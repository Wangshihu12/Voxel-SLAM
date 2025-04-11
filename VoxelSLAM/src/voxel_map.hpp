#ifndef VOXEL_MAP2_HPP
#define VOXEL_MAP2_HPP

#include "tools.hpp"
#include "preintegration.hpp"
#include <thread>
#include <Eigen/Eigenvalues>
#include <unordered_set>
#include <mutex>

#include <ros/ros.h>
#include <fstream>

/**
 * 带协方差矩阵的三维点结构体
 * 该结构体存储3D点的位置及其不确定性信息，是体素SLAM系统中的基本数据单元
 */
struct pointVar
{
  // 确保内存对齐，优化Eigen对象的性能
  // 这宏确保结构体的对象在内存中正确对齐，避免非对齐内存访问导致的性能下降
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  // 点的三维坐标
  // 使用Eigen库的Vector3d类型表示(x,y,z)坐标
  // 在体素SLAM中用于存储点的空间位置信息
  Eigen::Vector3d pnt;
  
  // 点的协方差矩阵
  // 3x3矩阵表示点在三个维度上的不确定性和相关性
  // 对角线元素表示x/y/z方向的方差，非对角线元素表示维度间的协方差
  // 在基于概率的SLAM系统中用于表示点位置的不确定性，影响优化权重
  Eigen::Matrix3d var;
};

using PVec = vector<pointVar>;
using PVecPtr = shared_ptr<vector<pointVar>>;

void down_sampling_pvec(PVec &pvec, double voxel_size, pcl::PointCloud<PointType> &pl_keep)
{
  unordered_map<VOXEL_LOC, pair<pointVar, int>> feat_map;
  float loc_xyz[3];
  for (pointVar &pv : pvec)
  {
    for (int j = 0; j < 3; j++)
    {
      loc_xyz[j] = pv.pnt[j] / voxel_size;
      if (loc_xyz[j] < 0)
        loc_xyz[j] -= 1.0;
    }

    VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter == feat_map.end())
    {
      feat_map[position] = make_pair(pv, 1);
    }
    else
    {
      pair<pointVar, int> &pp = iter->second;
      pp.first.pnt = (pp.first.pnt * pp.second + pv.pnt) / (pp.second + 1);
      pp.first.var = (pp.first.var * pp.second + pv.var) / (pp.second + 1);
      pp.second += 1;
    }
  }

  pcl::PointCloud<PointType>().swap(pl_keep);
  pl_keep.reserve(feat_map.size());
  PointType ap;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter)
  {
    pointVar &pv = iter->second.first;
    ap.x = pv.pnt[0];
    ap.y = pv.pnt[1];
    ap.z = pv.pnt[2];
    ap.normal_x = pv.var(0, 0);
    ap.normal_y = pv.var(1, 1);
    ap.normal_z = pv.var(2, 2);
    pl_keep.push_back(ap);
  }
}

struct Plane
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  Eigen::Matrix<double, 6, 6> plane_var;
  float radius = 0;
  bool is_plane = false;

  Plane()
  {
    plane_var.setZero();
  }
};

Eigen::Vector4d min_point;
double min_eigen_value;
int max_layer = 2;
int max_points = 100;
double voxel_size = 1.0;
int min_ba_point = 20;
vector<double> plane_eigen_value_thre;

void Bf_var(const pointVar &pv, Eigen::Matrix<double, 9, 9> &bcov, const Eigen::Vector3d &vec)
{
  Eigen::Matrix<double, 6, 3> Bi;
  // Eigen::Vector3d &vec = pv.world;
  Bi << 2 * vec(0), 0, 0,
      vec(1), vec(0), 0,
      vec(2), 0, vec(0),
      0, 2 * vec(1), 0,
      0, vec(2), vec(1),
      0, 0, 2 * vec(2);
  Eigen::Matrix<double, 6, 3> Biup = Bi * pv.var;
  bcov.block<6, 6>(0, 0) = Biup * Bi.transpose();
  bcov.block<6, 3>(0, 6) = Biup;
  bcov.block<3, 6>(6, 0) = Biup.transpose();
  bcov.block<3, 3>(6, 6) = pv.var;
}

// The LiDAR BA factor in optimization
class LidarFactor
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  vector<PointCluster> sig_vecs; // 每个体素的固定部分点集统计量
  vector<vector<PointCluster>> plvec_voxels; // 每个体素中各帧点集统计量
  vector<double> coeffs; // 每个体素的权重系数
  PLV(3) eig_values; // 特征值向量，PLV是向量类型的宏定义
  PLM(3) eig_vectors; // 特征向量矩阵，PLM是矩阵类型的宏定义
  vector<PointCluster> pcr_adds; // 每个体素的累积点集统计量
  int win_size; // 滑动窗口大小

  LidarFactor(int _w) : win_size(_w) {} // 构造函数，初始化窗口大小

  // 添加体素特征
  void push_voxel(vector<PointCluster> &vec_orig, PointCluster &fix, double coe, Eigen::Vector3d &eig_value, Eigen::Matrix3d &eig_vector, PointCluster &pcr_add)
  {
    plvec_voxels.push_back(vec_orig); // 添加各帧点集
    sig_vecs.push_back(fix); // 添加固定部分
    coeffs.push_back(coe); // 添加权重系数
    eig_values.push_back(eig_value); // 添加特征值
    eig_vectors.push_back(eig_vector); // 添加特征向量
    pcr_adds.push_back(pcr_add); // 添加累积统计量
  }

  // 计算点云特征对应的残差、雅可比和海森矩阵，用于非线性优化
  void acc_evaluate2(const vector<IMUST> &xs, int head, int end, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, double &residual)
  {
    Hess.setZero(); // 初始化海森矩阵为零
    JacT.setZero(); // 初始化雅可比向量为零
    residual = 0; // 初始化残差为零
    vector<PointCluster> sig_tran(win_size); // 创建变换后的点集数组
    const int kk = 0; // 使用第一个特征值（最小特征值）

    // 创建临时向量和矩阵数组
    PLV(3) viRiTuk(win_size); // 存储中间计算结果：vi×(Ri^T×uk)
    PLM(3) viRiTukukT(win_size); // 存储中间计算结果：viRiTuk×uk^T

    // 创建雅可比矩阵数组
    vector<Eigen::Matrix<double, 3, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 3, 6>>> Auk(win_size);
    Eigen::Matrix3d umumT; // 特征向量投影矩阵

    // 遍历分配给当前线程的体素
    for (int a = head; a < end; a++)
    {
      vector<PointCluster> &sig_orig = plvec_voxels[a]; // 获取当前体素的点集
      double coe = coeffs[a]; // 获取权重系数

      // PointCluster sig = sig_vecs[a];
      // for(int i=0; i<win_size; i++)
      // if(sig_orig[i].N != 0)
      // {
      //   sig_tran[i].transform(sig_orig[i], xs[i]);
      //   sig += sig_tran[i];
      // }

      // const Eigen::Vector3d &vBar = sig.v / sig.N;
      // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P/sig.N - vBar * vBar.transpose());
      // const Eigen::Vector3d &lmbd = saes.eigenvalues();
      // const Eigen::Matrix3d &U = saes.eigenvectors();
      // int NN = sig.N;

      // 获取预计算的特征值、特征向量和统计量
      Eigen::Vector3d lmbd = eig_values[a]; // 特征值
      Eigen::Matrix3d U = eig_vectors[a]; // 特征向量
      int NN = pcr_adds[a].N; // 点的总数量
      Eigen::Vector3d vBar = pcr_adds[a].v / NN; // 归一化质心

      // 提取特征向量
      Eigen::Vector3d u[3] = {U.col(0), U.col(1), U.col(2)};
      Eigen::Vector3d &uk = u[kk]; // 最小特征值对应的特征向量（法向量）
      Eigen::Matrix3d ukukT = uk * uk.transpose(); // 法向量的外积

      // 计算投影矩阵（用于雅可比投影）
      umumT.setZero();
      for (int i = 0; i < 3; i++)
        if (i != kk)
          umumT += 2.0 / (lmbd[kk] - lmbd[i]) * u[i] * u[i].transpose();

      for (int i = 0; i < win_size; i++) // 遍历每一帧
        // for(int i=1; i<win_size; i++)
        if (sig_orig[i].N != 0) // 如果当前帧在此体素中有点
        {
          // 提取当前帧在此体素的统计量
          Eigen::Matrix3d Pi = sig_orig[i].P; // 点的协方差矩阵
          Eigen::Vector3d vi = sig_orig[i].v; // 点的质心
          Eigen::Matrix3d Ri = xs[i].R; // 帧的旋转矩阵
          double ni = sig_orig[i].N; // 点的数量

          // 计算中间变量
          Eigen::Matrix3d vihat;
          vihat << SKEW_SYM_MATRX(vi); // vi的反对称矩阵
          Eigen::Vector3d RiTuk = Ri.transpose() * uk; // 旋转矩阵的转置乘以法向量
          Eigen::Matrix3d RiTukhat;
          RiTukhat << SKEW_SYM_MATRX(RiTuk); // RiTuk的反对称矩阵

          Eigen::Vector3d PiRiTuk = Pi * RiTuk;
          viRiTuk[i] = vihat * RiTuk;
          viRiTukukT[i] = viRiTuk[i] * uk.transpose();

          Eigen::Vector3d ti_v = xs[i].p - vBar; // 位置相对于质心的偏移
          double ukTti_v = uk.dot(ti_v);

          // 构建雅可比矩阵（关于旋转和平移的导数）
          Eigen::Matrix3d combo1 = hat(PiRiTuk) + vihat * ukTti_v;
          Eigen::Vector3d combo2 = Ri * vi + ni * ti_v;
          Auk[i].block<3, 3>(0, 0) = (Ri * Pi + ti_v * vi.transpose()) * RiTukhat - Ri * combo1;
          Auk[i].block<3, 3>(0, 3) = combo2 * uk.transpose() + combo2.dot(uk) * I33;
          Auk[i] /= NN;

          // 计算雅可比向量
          const Eigen::Matrix<double, 6, 1> &jjt = Auk[i].transpose() * uk;
          JacT.block<6, 1>(6 * i, 0) += coe * jjt;

          // 构建海森矩阵对角块（关于同一帧的二阶导数）
          const Eigen::Matrix3d &HRt = 2.0 / NN * (1.0 - ni / NN) * viRiTukukT[i];
          Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[i];
          // 旋转-旋转块
          Hb.block<3, 3>(0, 0) += 2.0 / NN * (combo1 - RiTukhat * Pi) * RiTukhat - 2.0 / NN / NN * viRiTuk[i] * viRiTuk[i].transpose() - 0.5 * hat(jjt.block<3, 1>(0, 0));
          // 旋转-平移块   
          Hb.block<3, 3>(0, 3) += HRt;
          // 平移-旋转块
          Hb.block<3, 3>(3, 0) += HRt.transpose();
          // 平移-平移块
          Hb.block<3, 3>(3, 3) += 2.0 / NN * (ni - ni * ni / NN) * ukukT;

          // 添加到全局海森矩阵
          Hess.block<6, 6>(6 * i, 6 * i) += coe * Hb;
        }

      // 计算交叉块（不同帧之间的关系）
      for (int i = 0; i < win_size - 1; i++)
        // for(int i=1; i<win_size-1; i++)
        if (sig_orig[i].N != 0)
        {
          double ni = sig_orig[i].N;
          for (int j = i + 1; j < win_size; j++)
            if (sig_orig[j].N != 0)
            {
              double nj = sig_orig[j].N;
              Eigen::Matrix<double, 6, 6> Hb = Auk[i].transpose() * umumT * Auk[j];
              Hb.block<3, 3>(0, 0) += -2.0 / NN / NN * viRiTuk[i] * viRiTuk[j].transpose();
              Hb.block<3, 3>(0, 3) += -2.0 * nj / NN / NN * viRiTukukT[i];
              Hb.block<3, 3>(3, 0) += -2.0 * ni / NN / NN * viRiTukukT[j].transpose();
              Hb.block<3, 3>(3, 3) += -2.0 * ni * nj / NN / NN * ukukT;

              Hess.block<6, 6>(6 * i, 6 * j) += coe * Hb;
            }
        }

      // 累加残差（使用最小特征值作为残差）
      residual += coe * lmbd[kk];
    }

    // 利用海森矩阵的对称性填充下三角部分
    for (int i = 1; i < win_size; i++)
      for (int j = 0; j < i; j++)
        Hess.block<6, 6>(6 * i, 6 * j) = Hess.block<6, 6>(6 * j, 6 * i).transpose();
  }

  // 只计算残差
  void evaluate_only_residual(const vector<IMUST> &xs, int head, int end, double &residual)
  {
    residual = 0;
    // vector<PointCluster> sig_tran(win_size);
    int kk = 0; // 使用第一个特征值（最小特征值）

    // int gps_size = plvec_voxels.size();
    PointCluster pcr;

    for (int a = head; a < end; a++) // 遍历分配给当前线程的体素
    {
      const vector<PointCluster> &sig_orig = plvec_voxels[a];
      PointCluster sig = sig_vecs[a]; // 固定部分点集

      // 变换并累加各帧点集
      for (int i = 0; i < win_size; i++)
        if (sig_orig[i].N != 0)
        {
          pcr.transform(sig_orig[i], xs[i]); // 应用位姿变换
          sig += pcr; // 累加到总点集
        }

      // 计算点集统计特性
      Eigen::Vector3d vBar = sig.v / sig.N; // 归一化质心
      // Eigen::Matrix3d cmt = sig.P/sig.N - vBar * vBar.transpose();
      // Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P - sig.v * vBar.transpose());
      // 计算点云协方差矩阵的特征分解
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sig.P / sig.N - vBar * vBar.transpose());
      Eigen::Vector3d lmbd = saes.eigenvalues();

      // centers[a] = vBar;
      // 更新预计算值，供后续使用
      eig_values[a] = saes.eigenvalues();
      eig_vectors[a] = saes.eigenvectors();
      pcr_adds[a] = sig;
      // Ns[a] = sig.N;

      // 累加残差（使用最小特征值作为残差）
      residual += coeffs[a] * lmbd[kk];
    }
  }

  // 清空所有数据
  void clear()
  {
    sig_vecs.clear(); // 清空固定部分点集
    plvec_voxels.clear(); // 清空各帧点集
    eig_values.clear(); // 清空特征值
    eig_vectors.clear(); // 清空特征向量
    pcr_adds.clear(); // 清空累积点集统计量
    coeffs.clear(); // 清空权重系数
  }

  ~LidarFactor() {}
};

// The LM optimizer for LiDAR BA
class Lidar_BA_Optimizer
{
public:
  int win_size, jac_leng, thd_num = 2;

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    // int thd_num = 4;
    double residual = 0;
    Hess.setZero();
    JacT.setZero();
    PLM(-1)
    hessians(thd_num);
    PLV(-1)
    jacobins(thd_num);

    for (int i = 0; i < thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    vector<double> resis(tthd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if (g_size < tthd_num)
      tthd_num = 1;

    vector<thread *> mthreads(tthd_num);
    double part = 1.0 * g_size / tthd_num;
    // for(int i=0; i<tthd_num; i++)
    for (int i = 1; i < tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part * i, part * (i + 1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    for (int i = 0; i < tthd_num; i++)
    {
      if (i != 0)
        mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      Hess += hessians[i];
      JacT += jacobins[i];
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess)
  {
    double residual1 = 0;
    // voxhess.evaluate_only_residual(x_stats, 0, voxhess.plvec_voxels.size(), residual1);

    // int thd_num = 2;
    vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if (g_size < thd_num)
    {
      printf("Too Less Voxel");
      exit(0);
    }
    vector<thread *> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for (int i = 1; i < thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part * i, part * (i + 1), ref(residuals[i]));

    for (int i = 0; i < thd_num; i++)
    {
      if (i != 0)
        mthreads[i]->join();
      else
        voxhess.evaluate_only_residual(x_stats, part * i, part * (i + 1), residuals[i]);
      residual1 += residuals[i];
      delete mthreads[i];
    }

    return residual1;
  }

  bool damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, Eigen::MatrixXd *hess, vector<double> &resis, int max_iter = 3, bool is_display = false)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;

    double u = 0.01, v = 2;
    Eigen::MatrixXd D(jac_leng, jac_leng), Hess(jac_leng, jac_leng);
    Eigen::VectorXd JacT(jac_leng), dxi(jac_leng);
    hess->resize(jac_leng, jac_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    bool is_converge = true;

    // double tt1 = ros::Time::now().toSec();
    // for(int i=0; i<10; i++)
    for (int i = 0; i < max_iter; i++)
    {
      if (is_calc_hess)
      {
        residual1 = divide_thread(x_stats, voxhess, Hess, JacT);
        *hess = Hess;
      }

      if (i == 0)
        resis.push_back(residual1);

      Hess.topRows(6).setZero();
      Hess.leftCols(6).setZero();
      Hess.block<6, 6>(0, 0).setIdentity();
      JacT.head(6).setZero();

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u * D).ldlt().solve(-JacT);

      for (int j = 0; j < win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(6 * j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(6 * j + 3, 0);
      }
      double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);

      residual2 = only_residual(x_stats_temp, voxhess);

      q = (residual1 - residual2);
      if (is_display)
        printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q / q1, q1, q);

      if (q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2 * q - 1, 3);
        u *= (q < one_three ? one_three : q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;
        is_converge = false;
      }

      if (fabs((residual1 - residual2) / residual1) < 1e-6)
        break;
    }
    resis.push_back(residual2);
    return is_converge;
  }
};

double imu_coef = 1e-4;
// double imu_coef = 1e-8;
#define DVEL 6
// The LiDAR-Inertial BA optimizer
class LI_BA_Optimizer
{
public:
  int win_size, jac_leng, imu_leng;

  void hess_plus(Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, Eigen::MatrixXd &hs, Eigen::VectorXd &js)
  {
    for (int i = 0; i < win_size; i++)
    {
      JacT.block<DVEL, 1>(i * DIM, 0) += js.block<DVEL, 1>(i * DVEL, 0);
      for (int j = 0; j < win_size; j++)
        Hess.block<DVEL, DVEL>(i * DIM, j * DIM) += hs.block<DVEL, DVEL>(i * DVEL, j * DVEL);
    }
  }

  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE *> &imus_factor, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    int thd_num = 5;
    double residual = 0;
    Hess.setZero();
    JacT.setZero();
    PLM(-1)
    hessians(thd_num);
    PLV(-1)
    jacobins(thd_num);
    vector<double> resis(thd_num, 0);

    for (int i = 0; i < thd_num; i++)
    {
      hessians[i].resize(jac_leng, jac_leng);
      jacobins[i].resize(jac_leng);
    }

    int tthd_num = thd_num;
    int g_size = voxhess.plvec_voxels.size();
    if (g_size < tthd_num)
      tthd_num = 1;
    double part = 1.0 * g_size / tthd_num;

    vector<thread *> mthreads(tthd_num);
    // for(int i=0; i<tthd_num; i++)
    for (int i = 1; i < tthd_num; i++)
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part * i, part * (i + 1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    Eigen::MatrixXd jtj(2 * DIM, 2 * DIM);
    Eigen::VectorXd gg(2 * DIM);

    for (int i = 0; i < win_size - 1; i++)
    {
      jtj.setZero();
      gg.setZero();
      residual += imus_factor[i]->give_evaluate(x_stats[i], x_stats[i + 1], jtj, gg, true);
      Hess.block<DIM * 2, DIM * 2>(i * DIM, i * DIM) += jtj;
      JacT.block<DIM * 2, 1>(i * DIM, 0) += gg;
    }

    Eigen::Matrix<double, DIM, DIM> joc;
    Eigen::Matrix<double, DIM, 1> rr;
    joc.setIdentity();
    rr.setZero();

    Hess *= imu_coef;
    JacT *= imu_coef;
    residual *= (imu_coef * 0.5);

    // printf("resi: %lf\n", residual);

    for (int i = 0; i < tthd_num; i++)
    {
      // mthreads[i]->join();
      if (i != 0)
        mthreads[i]->join();
      else
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]);
      hess_plus(Hess, JacT, hessians[i], jacobins[i]);
      residual += resis[i];
      delete mthreads[i];
    }

    return residual;
  }

  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE *> &imus_factor)
  {
    double residual1 = 0, residual2 = 0;
    Eigen::MatrixXd jtj(2 * DIM, 2 * DIM);
    Eigen::VectorXd gg(2 * DIM);

    int thd_num = 5;
    vector<double> residuals(thd_num, 0);
    int g_size = voxhess.plvec_voxels.size();
    if (g_size < thd_num)
    {
      // printf("Too Less Voxel"); exit(0);
      thd_num = 1;
    }
    vector<thread *> mthreads(thd_num, nullptr);
    double part = 1.0 * g_size / thd_num;
    for (int i = 1; i < thd_num; i++)
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part * i, part * (i + 1), ref(residuals[i]));

    for (int i = 0; i < win_size - 1; i++)
      residual1 += imus_factor[i]->give_evaluate(x_stats[i], x_stats[i + 1], jtj, gg, false);
    residual1 *= (imu_coef * 0.5);

    for (int i = 0; i < thd_num; i++)
    {
      if (i != 0)
      {
        mthreads[i]->join();
        delete mthreads[i];
      }
      else
        voxhess.evaluate_only_residual(x_stats, part * i, part * (i + 1), residuals[i]);
      residual2 += residuals[i];
    }

    return (residual1 + residual2);
  }

  void damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE *> &imus_factor, Eigen::MatrixXd *hess)
  {
    win_size = voxhess.win_size;
    jac_leng = win_size * 6;
    imu_leng = win_size * DIM;
    double u = 0.01, v = 2;
    Eigen::MatrixXd D(imu_leng, imu_leng), Hess(imu_leng, imu_leng);
    Eigen::VectorXd JacT(imu_leng), dxi(imu_leng);
    hess->resize(imu_leng, imu_leng);

    D.setIdentity();
    double residual1, residual2, q;
    bool is_calc_hess = true;
    vector<IMUST> x_stats_temp = x_stats;

    double hesstime = 0;
    double resitime = 0;

    // for(int i=0; i<10; i++)
    for (int i = 0; i < 3; i++)
    {
      if (is_calc_hess)
      {
        double tm = ros::Time::now().toSec();
        residual1 = divide_thread(x_stats, voxhess, imus_factor, Hess, JacT);
        hesstime += ros::Time::now().toSec() - tm;
        *hess = Hess;
      }

      Hess.topRows(DIM).setZero();
      Hess.leftCols(DIM).setZero();
      Hess.block<DIM, DIM>(0, 0).setIdentity();
      JacT.head(DIM).setZero();

      D.diagonal() = Hess.diagonal();
      dxi = (Hess + u * D).ldlt().solve(-JacT);

      for (int j = 0; j < win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DIM * j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DIM * j + 3, 0);
        x_stats_temp[j].v = x_stats[j].v + dxi.block<3, 1>(DIM * j + 6, 0);
        x_stats_temp[j].bg = x_stats[j].bg + dxi.block<3, 1>(DIM * j + 9, 0);
        x_stats_temp[j].ba = x_stats[j].ba + dxi.block<3, 1>(DIM * j + 12, 0);
      }

      for (int j = 0; j < win_size - 1; j++)
        imus_factor[j]->update_state(dxi.block<DIM, 1>(DIM * j, 0));

      double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);

      double tl1 = ros::Time::now().toSec();
      residual2 = only_residual(x_stats_temp, voxhess, imus_factor);
      double tl2 = ros::Time::now().toSec();
      // printf("onlyresi: %lf\n", tl2-tl1);
      resitime += tl2 - tl1;

      q = (residual1 - residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if (q > 0)
      {
        x_stats = x_stats_temp;
        double one_three = 1.0 / 3;

        q = q / q1;
        v = 2;
        q = 1 - pow(2 * q - 1, 3);
        u *= (q < one_three ? one_three : q);
        is_calc_hess = true;
      }
      else
      {
        u = u * v;
        v = 2 * v;
        is_calc_hess = false;

        for (int j = 0; j < win_size - 1; j++)
        {
          imus_factor[j]->dbg = imus_factor[j]->dbg_buf;
          imus_factor[j]->dba = imus_factor[j]->dba_buf;
        }
      }

      if (fabs((residual1 - residual2) / residual1) < 1e-6)
        break;
    }

    // printf("ba: %lf %lf %zu\n", hesstime, resitime, voxhess.plvec_voxels.size());
  }
};

// The LiDAR-Inertial BA optimizer with gravity optimization
class LI_BA_OptimizerGravity
{
public:
  int win_size, jac_leng, imu_leng;

  // Hess: 全局海森矩阵(Hessian Matrix)，存储整个系统的二阶导数信息，用于非线性优化
  // JacT: 全局雅可比矩阵的转置(Jacobian Transpose)，存储一阶导数信息
  // hs: 局部海森矩阵，表示当前处理的特征对海森矩阵的贡献
  // js: 局部雅可比矩阵转置，表示当前处理的特征对雅可比的贡献
  void hess_plus(Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT, Eigen::MatrixXd &hs, Eigen::VectorXd &js)
  {
    // 遍历滑动窗口中的每一帧
    for (int i = 0; i < win_size; i++)
    {
      // 更新雅可比向量：将局部雅可比的第i块加到全局雅可比的对应位置
      JacT.block<DVEL, 1>(i * DIM, 0) += js.block<DVEL, 1>(i * DVEL, 0);
      // 遍历与当前帧相关的所有帧
      for (int j = 0; j < win_size; j++)
        // 更新海森矩阵：将局部海森矩阵的(i,j)块加到全局海森矩阵的对应位置
        Hess.block<DVEL, DVEL>(i * DIM, j * DIM) += hs.block<DVEL, DVEL>(i * DVEL, j * DVEL);
    }
  }

  // 并行计算激光雷达和IMU因子，以构建完整的优化问题
  // x_stats: 位姿状态向量，包含滑动窗口中所有帧的位姿(IMUST类型)
  // voxhess: 激光雷达因子对象，存储了体素地图和点云特征
  // imus_factor: IMU预积分因子队列，包含相邻帧之间的IMU预积分结果
  // Hess: 海森矩阵，用于非线性优化的二阶导数信息
  // JacT: 雅可比矩阵的转置，用于非线性优化的一阶导数信息
  double divide_thread(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE *> &imus_factor, Eigen::MatrixXd &Hess, Eigen::VectorXd &JacT)
  {
    int thd_num = 5; // 设置线程数量为5
    double residual = 0; // 初始化残差值为0
    Hess.setZero(); // 初始化海森矩阵为零矩阵
    JacT.setZero(); // 初始化雅可比矩阵为零向量
    PLM(-1) hessians(thd_num); // 创建线程局部海森矩阵数组，PLM是宏定义，相当于vector<Eigen::MatrixXd>
    PLV(-1) jacobins(thd_num); // 创建线程局部雅可比向量数组，PLV是宏定义，相当于vector<Eigen::VectorXd>
    vector<double> resis(thd_num, 0); // 创建线程局部残差值数组，初始化为0

    // 分配线程局部内存
    for (int i = 0; i < thd_num; i++) // 遍历每个线程
    {
      hessians[i].resize(jac_leng, jac_leng); // 调整每个局部海森矩阵的大小
      jacobins[i].resize(jac_leng); // 调整每个局部雅可比向量的大小
    }

    int tthd_num = thd_num; // 初始计划使用的线程数
    int g_size = voxhess.plvec_voxels.size(); // 获取体素特征的总数量
    if (g_size < tthd_num)
      tthd_num = 1; // 如果特征数量少于线程数，只使用一个线程
    double part = 1.0 * g_size / tthd_num; // 计算每个线程处理的特征数量

    vector<thread *> mthreads(tthd_num); // 创建线程指针数组
    // for(int i=0; i<tthd_num; i++)
    for (int i = 1; i < tthd_num; i++) // 只为索引1及以上的线程创建新线程
      mthreads[i] = new thread(&LidarFactor::acc_evaluate2, &voxhess, x_stats, part * i, part * (i + 1), ref(hessians[i]), ref(jacobins[i]), ref(resis[i]));

    // 计算IMU因子贡献
    Eigen::MatrixXd jtj(2 * DIM + 3, 2 * DIM + 3); // 创建临时海森矩阵
    Eigen::VectorXd gg(2 * DIM + 3); // 创建临时雅可比向量

    // 遍历所有相邻帧对
    for (int i = 0; i < win_size - 1; i++)
    {
      jtj.setZero(); // 重置临时海森矩阵
      gg.setZero(); // 重置临时雅可比向量

      // 评估IMU因子，计算残差、海森矩阵和雅可比向量
      residual += imus_factor[i]->give_evaluate_g(x_stats[i], x_stats[i + 1], jtj, gg, true);

      // 将IMU因子贡献加到全局海森矩阵的对应位置
      Hess.block<DIM * 2, DIM * 2>(i * DIM, i * DIM) += jtj.block<2 * DIM, 2 * DIM>(0, 0); // 位姿对位姿的影响
      Hess.block<DIM * 2, 3>(i * DIM, imu_leng - 3) += jtj.block<2 * DIM, 3>(0, 2 * DIM); // 位姿对重力的影响
      Hess.block<3, DIM * 2>(imu_leng - 3, i * DIM) += jtj.block<3, 2 * DIM>(2 * DIM, 0); // 重力对位姿的影响
      Hess.block<3, 3>(imu_leng - 3, imu_leng - 3) += jtj.block<3, 3>(2 * DIM, 2 * DIM); // 重力对重力的影响

      // 将IMU因子雅可比贡献加到全局雅可比向量
      JacT.block<DIM * 2, 1>(i * DIM, 0) += gg.head(2 * DIM);
      JacT.tail(3) += gg.tail(3);
    }

    Eigen::Matrix<double, DIM, DIM> joc; // 创建先验信息矩阵
    Eigen::Matrix<double, DIM, 1> rr; // 创建先验残差向量
    joc.setIdentity(); // 设为单位矩阵
    rr.setZero(); // 设为零向量

    Hess *= imu_coef; // 对海森矩阵应用IMU权重系数
    JacT *= imu_coef; // 对雅可比向量应用IMU权重系数
    residual *= (imu_coef * 0.5); // 对残差应用IMU权重系数（乘0.5是因为残差是二次型）

    // printf("resi: %lf\n", residual);

    // 等待线程完成并合并结果
    for (int i = 0; i < tthd_num; i++) // 遍历所有线程
    {
      // mthreads[i]->join();
      if (i != 0) // 对于非主线程
        mthreads[i]->join(); // 等待线程完成
      else // 对于主线程(索引0)
        voxhess.acc_evaluate2(x_stats, 0, part, hessians[0], jacobins[0], resis[0]); // 在主线程中计算第一部分

      // 合并线程局部结果到全局结果
      hess_plus(Hess, JacT, hessians[i], jacobins[i]);
      residual += resis[i]; // 累加残差
      delete mthreads[i]; // 释放线程资源
    }

    return residual; // 返回总残差
  }

  // 计算当前状态下的优化残差值（不计算雅可比和海森矩阵）
  // x_stats: 位姿状态向量，包含滑动窗口中所有帧的位姿(IMUST类型)
  // voxhess: 激光雷达因子对象，存储了体素地图和点云特征
  // imus_factor: IMU预积分因子队列，包含相邻帧之间的IMU预积分结果
  double only_residual(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE *> &imus_factor)
  {
    double residual1 = 0, residual2 = 0; // 初始化IMU残差和激光雷达残差
    Eigen::MatrixXd jtj(2 * DIM, 2 * DIM); // 创建临时海森矩阵（仅用于IMU评估接口）
    Eigen::VectorXd gg(2 * DIM); // 创建临时雅可比向量（仅用于IMU评估接口）

    int thd_num = 5; // 设置线程数量为5
    vector<double> residuals(thd_num, 0); // 创建线程局部残差值数组，初始化为0
    int g_size = voxhess.plvec_voxels.size(); // 获取体素特征的总数量
    if (g_size < thd_num)
    {
      // printf("Too Less Voxel"); exit(0);
      thd_num = 1; // 如果特征数量少于线程数，只使用一个线程
    }

    vector<thread *> mthreads(thd_num, nullptr); // 创建线程指针数组，初始化为nullptr
    double part = 1.0 * g_size / thd_num; // 计算每个线程处理的特征数量
    for (int i = 1; i < thd_num; i++) // 只为索引1及以上的线程创建新线程
      mthreads[i] = new thread(&LidarFactor::evaluate_only_residual, &voxhess, x_stats, part * i, part * (i + 1), ref(residuals[i]));

    for (int i = 0; i < win_size - 1; i++) // 遍历滑动窗口中的所有相邻帧对
      // 评估IMU因子的残差值，最后一个参数false表示只计算残差，不计算雅可比和海森矩阵
      residual1 += imus_factor[i]->give_evaluate_g(x_stats[i], x_stats[i + 1], jtj, gg, false);
    residual1 *= (imu_coef * 0.5); // 应用IMU残差权重系数

    for (int i = 0; i < thd_num; i++) // 遍历所有线程
    {
      if (i != 0) // 对于非主线程
      {
        mthreads[i]->join(); // 等待线程完成
        delete mthreads[i]; // 释放线程资源
      }
      else // 对于主线程(索引0)
        // 在主线程中计算第一部分激光雷达残差
        voxhess.evaluate_only_residual(x_stats, part * i, part * (i + 1), residuals[i]);
      residual2 += residuals[i]; // 累加激光雷达残差
    }

    return (residual1 + residual2); // 返回总残差（IMU + 激光雷达）
  }

  // x_stats: 位姿状态向量，包含滑动窗口中所有帧的位姿，函数将更新这些状态
  // voxhess: 激光雷达因子，提供点云特征约束
  // imus_factor: IMU预积分因子队列，提供相邻帧之间的运动约束
  // resis: 残差值向量，用于记录优化前后的残差变化
  // hess: 指向海森矩阵的指针，用于保存最终的海森矩阵（可用于不确定性分析）
  // max_iter: 最大迭代次数，默认为2
  void damping_iter(vector<IMUST> &x_stats, LidarFactor &voxhess, deque<IMU_PRE *> &imus_factor, vector<double> &resis, Eigen::MatrixXd *hess, int max_iter = 2)
  {
    win_size = voxhess.win_size; // 获取滑动窗口大小
    jac_leng = win_size * 6; // 计算雅可比向量长度（每帧6自由度）
    imu_leng = win_size * DIM + 3; // 计算总参数长度（含位姿和重力）
    double u = 0.01, v = 2; // 初始化LM算法参数：阻尼因子u和增长系数v
    Eigen::MatrixXd D(imu_leng, imu_leng), Hess(imu_leng, imu_leng); // 创建对角矩阵D和海森矩阵
    Eigen::VectorXd JacT(imu_leng), dxi(imu_leng); // 创建雅可比向量和增量向量

    D.setIdentity(); // 初始化对角矩阵为单位矩阵
    double residual1, residual2, q; // 优化前后残差和实际改进量
    bool is_calc_hess = true; // 标记是否需要计算海森矩阵（优化策略）
    vector<IMUST> x_stats_temp = x_stats; // 创建状态临时副本，用于尝试更新

    // 优化迭代主循环
    for (int i = 0; i < max_iter; i++)
    {
      if (is_calc_hess) // 如果需要计算海森矩阵
      {
        // 使用多线程计算当前状态下的海森矩阵、雅可比向量和残差
        residual1 = divide_thread(x_stats, voxhess, imus_factor, Hess, JacT);
        *hess = Hess; // 保存海森矩阵
      }

      if (i == 0) // 第一次迭代
        resis.push_back(residual1); // 记录初始残差

      // 固定滑动窗口中第一帧的位姿（作为参考坐标系）
      Hess.topRows(6).setZero();
      Hess.leftCols(6).setZero();
      Hess.block<6, 6>(0, 0).setIdentity();
      JacT.head(6).setZero();

      // Hess.rightCols(3).setZero();
      // Hess.bottomRows(3).setZero();
      // Hess.block<3, 3>(imu_leng-3, imu_leng-3).setIdentity();
      // JacT.tail(3).setZero();

      D.diagonal() = Hess.diagonal(); // 设置对角矩阵D为海森矩阵的对角线
      // 求解LM线性方程，得到增量 dxi
      dxi = (Hess + u * D).ldlt().solve(-JacT);

      // 更新重力向量，所有帧共享同一重力向量
      x_stats_temp[0].g += dxi.tail(3);
      // 更新滑动窗口中每一帧的状态
      for (int j = 0; j < win_size; j++)
      {
        x_stats_temp[j].R = x_stats[j].R * Exp(dxi.block<3, 1>(DIM * j, 0));
        x_stats_temp[j].p = x_stats[j].p + dxi.block<3, 1>(DIM * j + 3, 0);
        x_stats_temp[j].v = x_stats[j].v + dxi.block<3, 1>(DIM * j + 6, 0);
        x_stats_temp[j].bg = x_stats[j].bg + dxi.block<3, 1>(DIM * j + 9, 0);
        x_stats_temp[j].ba = x_stats[j].ba + dxi.block<3, 1>(DIM * j + 12, 0);
        x_stats_temp[j].g = x_stats_temp[0].g;
      }

      // 更新IMU预积分中的状态
      for (int j = 0; j < win_size - 1; j++)
        imus_factor[j]->update_state(dxi.block<DIM, 1>(DIM * j, 0));

      // 计算预测的残差减少量
      double q1 = 0.5 * dxi.dot(u * D * dxi - JacT);
      // 计算实际的新残差
      residual2 = only_residual(x_stats_temp, voxhess, imus_factor);
      // 计算实际残差减少量
      q = (residual1 - residual2);
      // printf("iter%d: (%lf %lf) u: %lf v: %.1lf q: %.2lf %lf %lf\n", i, residual1, residual2, u, v, q/q1, q1, q);

      if (q > 0) // 如果残差确实减少了
      {
        x_stats = x_stats_temp; // 接受新状态
        double one_three = 1.0 / 3;

        q = q / q1; // 实际减少量与预测减少量的比值
        v = 2; // 重置增长系数
        q = 1 - pow(2 * q - 1, 3); // 计算阻尼因子调整比例
        u *= (q < one_three ? one_three : q); // 减小阻尼因子（最小为原来的1/3）
        is_calc_hess = true; // 下次迭代需要重新计算海森矩阵
      }
      else // 如果残差增加了
      {
        u = u * v; // 增大阻尼因子
        v = 2 * v; // 增大增长系数
        is_calc_hess = false; // 下次迭代不需要重新计算海森矩阵

        // 恢复IMU偏置增量为缓存值
        for (int j = 0; j < win_size - 1; j++)
        {
          imus_factor[j]->dbg = imus_factor[j]->dbg_buf;
          imus_factor[j]->dba = imus_factor[j]->dba_buf;
        }
      }

      // 收敛性检查：如果相对残差减少量小于阈值，提前结束迭代
      if (fabs((residual1 - residual2) / residual1) < 1e-6)
        break;
    }
    resis.push_back(residual2); // 记录最终残差
  }
};

// 10 scans merge into a keyframe
struct Keyframe
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  IMUST x0;
  pcl::PointCloud<PointType>::Ptr plptr;
  int exist;
  int id, mp;
  float jour;

  Keyframe(IMUST &_x0) : x0(_x0), exist(0)
  {
    plptr.reset(new pcl::PointCloud<PointType>());
  }

  void generate(pcl::PointCloud<PointType> &pl_send, Eigen::Matrix3d rot = Eigen::Matrix3d::Identity(), Eigen::Vector3d tra = Eigen::Vector3d(0, 0, 0))
  {
    Eigen::Vector3d v3;
    for (PointType ap : plptr->points)
    {
      v3 << ap.x, ap.y, ap.z;
      v3 = rot * v3 + tra;
      ap.x = v3[0];
      ap.y = v3[1];
      ap.z = v3[2];
      pl_send.push_back(ap);
    }
  }
};

// The sldingwindow in each voxel nodes
class SlideWindow
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  vector<PVec> points;
  vector<PointCluster> pcrs_local;

  SlideWindow(int wdsize)
  {
    pcrs_local.resize(wdsize);
    points.resize(wdsize);
    for (int i = 0; i < wdsize; i++)
      points[i].reserve(20);
  }

  void resize(int wdsize)
  {
    if (points.size() != wdsize)
    {
      points.resize(wdsize);
      pcrs_local.resize(wdsize);
    }
  }

  void clear()
  {
    int wdsize = points.size();
    for (int i = 0; i < wdsize; i++)
    {
      points[i].clear();
      pcrs_local[i].clear();
    }
  }
};

// The octotree map for odometry and local mapping
// You can re-write it in your own project
int *mp;
class OctoTree
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SlideWindow *sw = nullptr; // 滑动窗口指针,用于存储时间窗口内的点云数据
  PointCluster pcr_add; // 当前节点中添加的点云的集合统计
  Eigen::Matrix<double, 9, 9> cov_add; // 协方差矩阵(9x9),用于平面参数估计

  PointCluster pcr_fix; // 固定点云的集合统计,不随位姿优化变化
  PVec point_fix; // 固定点的向量,包含点和协方差

  // 当前节点在八叉树中的层级
  int layer;
  // 八叉树节点状态:0表示叶子节点,1表示内部节点
  int octo_state;
  // 滑动窗口大小
  int wdsize;
  OctoTree *leaves[8]; // 八个子节点指针数组
  double voxel_center[3]; // 体素中心坐标
  double jour = 0; // 时间戳或序号,用于跟踪节点的创建时间
  float quater_length; // 体素的四分之一边长

  Plane plane; // 节点中提取的平面特征
  bool isexist = false; // 节点是否存在有效数据

  Eigen::Vector3d eig_value; // 特征值(用于平面判断)
  Eigen::Matrix3d eig_vector; // 特征向量(平面的法向量在第一列)

  // 上次更新时的点数
  int last_num = 0;
  // 优化状态标志,-1表示未参与优化
  int opt_state = -1;
  mutex mVox;

  /**
   * 八叉树节点构造函数
   * @param _l 当前节点的层级
   * @param _w 滑动窗口大小
   */
  OctoTree(int _l, int _w) : layer(_l), wdsize(_w), octo_state(0)
  {
    // 初始化所有子节点指针为空
    for (int i = 0; i < 8; i++)
      leaves[i] = nullptr;
    // 初始化协方差矩阵为零矩阵
    cov_add.setZero();

    // 注释掉的代码: 曾用于生成随机颜色或标识值
    // ins = 255.0*rand()/(RAND_MAX + 1.0f);
  }

  /**
   * 将点添加到八叉树节点中的内联函数
   * @param ord 点在原始序列中的索引
   * @param pv 点的数据结构,包含位置和协方差
   * @param pw 点在世界坐标系中的位置
   * @param sws 可重用的滑动窗口指针列表
   */
  inline void push(int ord, const pointVar &pv, const Eigen::Vector3d &pw, vector<SlideWindow *> &sws)
  {
    // 加锁以保护多线程访问
    mVox.lock();
    // 如果当前节点没有滑动窗口,则创建或重用一个
    if (sw == nullptr)
    {
      if (sws.size() != 0)
      {
        // 从池中获取一个滑动窗口
        sw = sws.back();
        sws.pop_back();
        // 调整滑动窗口大小与当前节点匹配
        sw->resize(wdsize);
      }
      else
        // 没有可用的滑动窗口,创建一个新的
        sw = new SlideWindow(wdsize);
    }
    // 标记节点为有效数据
    if (!isexist)
      isexist = true;

    // 获取在映射数组中的索引
    int mord = mp[ord];
    // 如果节点层级小于最大层级,存储原始点数据
    if (layer < max_layer)
      sw->points[mord].push_back(pv);
    // 在滑动窗口中添加点的局部坐标
    sw->pcrs_local[mord].push(pv.pnt);
    // 在当前节点累积世界坐标系中的点
    pcr_add.push(pw);
    // 计算并累积协方差矩阵,用于平面拟合
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pw);
    cov_add += Bi;
    // 解锁
    mVox.unlock();
  }

  /**
   * 添加固定点到八叉树节点的内联函数
   * @param pv 要添加的固定点数据结构(包含位置和协方差)
   */
  inline void push_fix(pointVar &pv)
  {
    // 如果节点层级小于最大层级,存储原始点
    if (layer < max_layer)
      point_fix.push_back(pv);
    // 添加点到固定点云统计数据中
    pcr_fix.push(pv.pnt);
    // 同时累加到节点的总点云统计中
    pcr_add.push(pv.pnt);
    // 计算并累积协方差矩阵
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pv.pnt);
    cov_add += Bi;
  }

  /**
   * 添加固定点到八叉树节点的内联函数(不计算协方差)
   * @param pv 要添加的固定点数据结构
   */
  inline void push_fix_novar(pointVar &pv)
  {
    // 如果节点层级小于最大层级,存储原始点
    if (layer < max_layer)
      point_fix.push_back(pv);
    // 添加点到固定点云统计数据中
    pcr_fix.push(pv.pnt);
    // 同时累加到节点的总点云统计中
    pcr_add.push(pv.pnt);
    // 注意: 与push_fix不同,这里不计算协方差矩阵
  }

  /**
   * 判断点集是否构成平面的内联函数
   * @param eig_values 点云协方差矩阵的特征值(按升序排列)
   * @return 如果满足平面条件返回true,否则返回false
   */
  inline bool plane_judge(Eigen::Vector3d &eig_values)
  {
    // 注释掉的简单判断: 仅检查最小特征值是否小于阈值
    // return (eig_values[0] < min_eigen_value);
    
    // 当前使用的判断条件:
    // 1. 最小特征值小于阈值 且
    // 2. 最小特征值与最大特征值的比值小于层级相关的阈值
    // 这两个条件共同确保点云分布是扁平的(平面状)
    return (eig_values[0] < min_eigen_value && (eig_values[0] / eig_values[2]) < plane_eigen_value_thre[layer]);
  }

  /**
   * 将点分配到八叉树中合适位置的函数
   * @param ord 点在原始序列中的索引
   * @param pv 点的数据结构,包含位置和协方差
   * @param pw 点在世界坐标系中的位置
   * @param sws 可重用的滑动窗口指针列表
   */
  void allocate(int ord, const pointVar &pv, const Eigen::Vector3d &pw, vector<SlideWindow *> &sws)
  {
    // 如果是叶子节点,直接添加点到当前节点
    if (octo_state == 0)
    {
      push(ord, pv, pw, sws);
    }
    else
    {
      // 根据点的世界坐标相对于当前节点中心的位置确定子节点
      int xyz[3] = {0, 0, 0};
      for (int k = 0; k < 3; k++)
        if (pw[k] > voxel_center[k])
          xyz[k] = 1;
      // 计算子节点索引(0-7)
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

      // 如果对应的子节点不存在,创建一个新的
      if (leaves[leafnum] == nullptr)
      {
        // 创建新的子节点,层级+1
        leaves[leafnum] = new OctoTree(layer + 1, wdsize);
        // 计算子节点的中心坐标
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
        // 子节点的四分之一长度是当前节点的一半
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      // 递归地将点分配到子节点中
      leaves[leafnum]->allocate(ord, pv, pw, sws);
    }
  }

  void allocate_fix(pointVar &pv)
  {
    if (octo_state == 0)
    {
      push_fix_novar(pv);
    }
    else if (layer < max_layer)
    {
      int xyz[3] = {0, 0, 0};
      for (int k = 0; k < 3; k++)
        if (pv.pnt[k] > voxel_center[k])
          xyz[k] = 1;
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

      if (leaves[leafnum] == nullptr)
      {
        leaves[leafnum] = new OctoTree(layer + 1, wdsize);
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      leaves[leafnum]->allocate_fix(pv);
    }
  }

  /**
   * 将固定点集合划分并分配到八叉树子节点中
   * @param sws 可重用的滑动窗口指针列表
   */
  void fix_divide(vector<SlideWindow *> &sws)
  {
    // 遍历当前节点中存储的所有固定点
    for (pointVar &pv : point_fix)
    {
      // 根据点相对于当前节点中心的位置确定其所属的子节点
      int xyz[3] = {0, 0, 0};
      for (int k = 0; k < 3; k++)
        if (pv.pnt[k] > voxel_center[k])
          xyz[k] = 1;
      // 计算子节点索引(0-7)
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      
      // 如果对应的子节点不存在,创建一个新的
      if (leaves[leafnum] == nullptr)
      {
        // 创建新的子节点,层级+1
        leaves[leafnum] = new OctoTree(layer + 1, wdsize);
        // 计算子节点的中心坐标
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
        // 子节点的四分之一长度是当前节点的一半
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      // 将固定点添加到对应的子节点中(使用带协方差计算的方式)
      leaves[leafnum]->push_fix(pv);
    }
  }

  /**
   * 将当前节点特定时间索引的点云重新分配到八叉树子节点中
   * @param si 时间索引(在滑动窗口索引数组中的位置)
   * @param xx IMU状态,包含位姿信息用于坐标变换
   * @param sws 可重用的滑动窗口指针列表
   */
  void subdivide(int si, IMUST &xx, vector<SlideWindow *> &sws)
  {
    // 遍历当前节点中特定时间索引的所有点
    for (pointVar &pv : sw->points[mp[si]])
    {
      // 将点从局部坐标转换到世界坐标系
      Eigen::Vector3d pw = xx.R * pv.pnt + xx.p;
      // 根据点的世界坐标相对于当前节点中心的位置确定子节点
      int xyz[3] = {0, 0, 0};
      for (int k = 0; k < 3; k++)
        if (pw[k] > voxel_center[k])
          xyz[k] = 1;
      // 计算子节点索引(0-7)
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      
      // 如果对应的子节点不存在,创建一个新的
      if (leaves[leafnum] == nullptr)
      {
        // 创建新的子节点,层级+1
        leaves[leafnum] = new OctoTree(layer + 1, wdsize);
        // 计算子节点的中心坐标
        leaves[leafnum]->voxel_center[0] = voxel_center[0] + (2 * xyz[0] - 1) * quater_length;
        leaves[leafnum]->voxel_center[1] = voxel_center[1] + (2 * xyz[1] - 1) * quater_length;
        leaves[leafnum]->voxel_center[2] = voxel_center[2] + (2 * xyz[2] - 1) * quater_length;
        // 子节点的四分之一长度是当前节点的一半
        leaves[leafnum]->quater_length = quater_length / 2;
      }

      // 将点添加到对应的子节点中
      leaves[leafnum]->push(si, pv, pw, sws);
    }
  }

  /**
   * 更新平面参数及其协方差矩阵
   * 该函数在检测到平面特征时调用,用于计算平面的精确参数和不确定性
   */
  void plane_update()
  {
    // 计算平面中心点(点云质心)
    plane.center = pcr_add.v / pcr_add.N;
    // 使用最小特征值对应的特征向量作为平面法向量
    int l = 0;
    // 提取所有特征向量构成正交基底
    Eigen::Vector3d u[3] = {eig_vector.col(0), eig_vector.col(1), eig_vector.col(2)};
    // 点数的倒数,用于归一化
    double nv = 1.0 / pcr_add.N;

    // 计算特征向量对协方差的导数
    Eigen::Matrix<double, 3, 9> u_c;
    u_c.setZero();
    for (int k = 0; k < 3; k++)
      if (k != l) // 对非法向量方向的特征向量
      {
        // 计算外积矩阵
        Eigen::Matrix3d ukl = u[k] * u[l].transpose();
        // 构建平面参数雅可比矩阵的一部分
        Eigen::Matrix<double, 1, 9> fkl;
        fkl.head(6) << ukl(0, 0), ukl(1, 0) + ukl(0, 1), ukl(2, 0) + ukl(0, 2),
            ukl(1, 1), ukl(1, 2) + ukl(2, 1), ukl(2, 2);
        // 计算与平面中心相关的部分
        fkl.tail(3) = -(u[k].dot(plane.center) * u[l] + u[l].dot(plane.center) * u[k]);

        // 累加每个非法向量方向的贡献,加权系数与特征值差异相关
        u_c += nv / (eig_value[l] - eig_value[k]) * u[k] * fkl;
      }

    // 计算平面参数协方差传播
    Eigen::Matrix<double, 3, 9> Jc = u_c * cov_add;
    // 法向量协方差
    plane.plane_var.block<3, 3>(0, 0) = Jc * u_c.transpose();
    // 法向量与中心点的协方差
    Eigen::Matrix3d Jc_N = nv * Jc.block<3, 3>(0, 6);
    plane.plane_var.block<3, 3>(0, 3) = Jc_N;
    plane.plane_var.block<3, 3>(3, 0) = Jc_N.transpose();
    // 中心点的协方差
    plane.plane_var.block<3, 3>(3, 3) = nv * nv * cov_add.block<3, 3>(6, 6);
    // 设置平面法向量为第一个特征向量(对应最小特征值)
    plane.normal = u[0];
    // 平面的半径用最大特征值表示(表示点云在平面内的展布程度)
    plane.radius = eig_value[2];
  }

  /**
   * 重新划分八叉树节点,基于点云分布特性决定是否需要细分
   * @param win_count 滑动窗口中的帧数
   * @param x_buf 存储各帧位姿的IMU状态缓冲区
   * @param sws 可重用的滑动窗口指针列表
   */
  void recut(int win_count, vector<IMUST> &x_buf, vector<SlideWindow *> &sws)
  {
    // 处理叶子节点
    if (octo_state == 0)
    {
      if (layer >= 0) // 忽略负层级(根节点上层)
      {
        // 重置优化状态标志
        opt_state = -1;
        // 如果点数太少,则不是平面
        if (pcr_add.N <= min_point[layer])
        {
          plane.is_plane = false;
          return;
        }
        // 如果节点无效或没有滑动窗口,直接返回
        if (!isexist || sw == nullptr)
          return;

        // 计算点云协方差矩阵的特征值和特征向量
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
        eig_value = saes.eigenvalues();
        eig_vector = saes.eigenvectors();
        // 判断是否满足平面条件
        plane.is_plane = plane_judge(eig_value);

        // 如果是平面特征,不需要进一步细分
        if (plane.is_plane)
        {
          return;
        }
        // 如果已达最大层级,也不再细分
        else if (layer >= max_layer)
          return;
      }

      // 处理固定点,将它们分配到子节点
      if (pcr_fix.N != 0)
      {
        fix_divide(sws);
        // 释放固定点数组内存(使用swap而不是clear以确保内存被释放)
        PVec().swap(point_fix);
      }

      // 将滑动窗口中每一帧的点云重新分配到子节点
      for (int i = 0; i < win_count; i++)
        subdivide(i, x_buf[i], sws);

      // 清空滑动窗口并将其回收到池中
      sw->clear();
      sws.push_back(sw);
      sw = nullptr;
      // 将当前节点状态改为内部节点
      octo_state = 1;
    }

    // 对所有非空子节点递归调用recut
    for (int i = 0; i < 8; i++)
      if (leaves[i] != nullptr)
        leaves[i]->recut(win_count, x_buf, sws);
  }

  /**
   * 执行滑动窗口边缘化操作,将早期帧点云转换为固定点云
   * @param win_count 滑动窗口中的总帧数
   * @param mgsize 要边缘化的帧数(窗口中最早的几帧)
   * @param x_buf 存储各帧位姿的IMU状态缓冲区
   * @param vox_opt 优化后的LiDAR因子,包含优化后的点云统计信息
   */
  void margi(int win_count, int mgsize, vector<IMUST> &x_buf, const LidarFactor &vox_opt)
  {
    // 处理叶子节点且非根节点
    if (octo_state == 0 && layer >= 0)
    {
      // 如果节点无效或没有滑动窗口,直接返回
      if (!isexist || sw == nullptr)
        return;
      // 加锁以保护多线程访问
      mVox.lock();
      // 创建临时数组存储转换到世界坐标系的点云
      vector<PointCluster> pcrs_world(wdsize);
      
      // 注释掉的代码: 直接累加世界坐标系下的点,现在通过优化后的结果处理
      // pcr_add = pcr_fix;
      // for(int i=0; i<win_count; i++)
      // if(sw->pcrs_local[mp[i]].N != 0)
      // {
      //   pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
      //   pcr_add += pcrs_world[i];
      // }

      // 检查优化状态索引是否有效
      if (opt_state >= int(vox_opt.pcr_adds.size()))
      {
        printf("Error: opt_state: %d %zu\n", opt_state, vox_opt.pcr_adds.size());
        exit(0);
      }

      // 如果节点参与了优化,使用优化后的结果
      if (opt_state >= 0)
      {
        // 从优化结果中获取点云统计、特征值和特征向量
        pcr_add = vox_opt.pcr_adds[opt_state];
        eig_value = vox_opt.eig_values[opt_state];
        eig_vector = vox_opt.eig_vectors[opt_state];
        opt_state = -1;

        // 仅转换要边缘化的帧,但不累加到pcr_add中
        for (int i = 0; i < mgsize; i++)
          if (sw->pcrs_local[mp[i]].N != 0)
          {
            pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
          }
      }
      else // 节点未参与优化,需要重新计算
      {
        // 从固定点云开始累加
        pcr_add = pcr_fix;
        // 转换滑动窗口中所有帧的点云到世界坐标系并累加
        for (int i = 0; i < win_count; i++)
          if (sw->pcrs_local[mp[i]].N != 0)
          {
            pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
            pcr_add += pcrs_world[i];
          }

        // 如果是平面特征,重新计算特征值和特征向量
        if (plane.is_plane)
        {
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
          eig_value = saes.eigenvalues();
          eig_vector = saes.eigenvectors();
        }
      }

      // 如果固定点数量未达到上限且是平面特征,考虑更新平面参数
      if (pcr_fix.N < max_points && plane.is_plane)
        // 当点数增加较多或初始点数较少时更新平面参数
        if (pcr_add.N - last_num >= 5 || last_num <= 10)
        {
          plane_update();
          last_num = pcr_add.N;
        }

      // 如果固定点数量未达到上限,将边缘化的点添加到固定点中
      if (pcr_fix.N < max_points)
      {
        for (int i = 0; i < mgsize; i++)
          if (pcrs_world[i].N != 0)
          {
            // 累加点云统计
            pcr_fix += pcrs_world[i];
            // 将每个点转换到世界坐标系并添加到固定点数组
            for (pointVar pv : sw->points[mp[i]])
            {
              pv.pnt = x_buf[i].R * pv.pnt + x_buf[i].p;
              point_fix.push_back(pv);
            }
          }
      }
      else // 固定点已达上限,不再添加新点
      {
        // 从累加结果中减去将被边缘化的点云
        for (int i = 0; i < mgsize; i++)
          if (pcrs_world[i].N != 0)
            pcr_add -= pcrs_world[i];

        // 清空固定点数组以节省内存
        if (point_fix.size() != 0)
          PVec().swap(point_fix);
      }

      // 清理边缘化帧在滑动窗口中的数据
      for (int i = 0; i < mgsize; i++)
        if (sw->pcrs_local[mp[i]].N != 0)
        {
          sw->pcrs_local[mp[i]].clear();
          sw->points[mp[i]].clear();
        }

      // 更新节点有效性状态:如果所有点都变成固定点,则标记为无效
      if (pcr_fix.N >= pcr_add.N)
        isexist = false;
      else
        isexist = true;

      // 解锁
      mVox.unlock();
    }
    else // 非叶子节点或根节点处理
    {
      // 默认标记为无效
      isexist = false;
      // 递归处理子节点
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
        {
          leaves[i]->margi(win_count, mgsize, x_buf, vox_opt);
          // 如果任何子节点有效,则当前节点也标记为有效
          isexist = isexist || leaves[i]->isexist;
        }
    }
  }

  /**
   * 提取用于LiDAR优化的因子
   * 遍历八叉树结构,找出符合条件的平面特征并添加到优化因子中
   * @param vox_opt LiDAR因子对象,用于收集各个体素的平面特征
   */
  void tras_opt(LidarFactor &vox_opt)
  {
    // 处理叶子节点
    if (octo_state == 0)
    {
      // 检查节点是否有效且包含平面特征
      if (layer >= 0 && isexist && plane.is_plane && sw != nullptr)
      {
        // 额外的平面质量检查:如果最小特征值与次小特征值比值大于0.12,
        // 说明平面不够"平",放弃使用该特征
        if (eig_value[0] / eig_value[1] > 0.12)
          return;

        // 设置权重系数为1(可以根据平面质量调整)
        double coe = 1;
        // 创建点云集合副本,用于优化
        vector<PointCluster> pcrs(wdsize);
        // 复制各帧在该节点中的局部点云
        for (int i = 0; i < wdsize; i++)
          pcrs[i] = sw->pcrs_local[mp[i]];
        // 记录该节点在优化因子中的索引,用于后续更新
        opt_state = vox_opt.plvec_voxels.size();
        // 将节点的点云数据添加到优化因子中
        vox_opt.push_voxel(pcrs, pcr_fix, coe, eig_value, eig_vector, pcr_add);
      }
    }
    else // 处理非叶子节点
    {
      // 递归地处理所有非空子节点
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
          leaves[i]->tras_opt(vox_opt);
    }
  }

  // 在八叉树节点中匹配点与平面
  // 参数:
  // wld - 待匹配点的世界坐标
  // pla - 匹配到的平面指针
  // max_prob - 最大匹配概率
  // var_wld - 点的协方差矩阵
  // sigma_d - 匹配的不确定度
  // oc - 匹配到的八叉树节点指针
  // 返回值: 1表示匹配成功,0表示匹配失败
  int match(Eigen::Vector3d &wld, Plane *&pla, double &max_prob, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree *&oc)
  {
    int flag = 0;
    if (octo_state == 0) // 如果是叶子节点
    {
      if (plane.is_plane) // 如果节点包含平面特征
      {
        // 计算点到平面的距离
        float dis_to_plane = fabs(plane.normal.dot(wld - plane.center));
        // 计算点到平面中心的距离
        float dis_to_center = (plane.center - wld).squaredNorm();
        // 计算点到平面的垂直距离
        float range_dis = (dis_to_center - dis_to_plane * dis_to_plane);
        
        // 判断点是否在平面的有效范围内(3倍标准差)
        if (range_dis <= 3 * 3 * plane.radius)
        {
          // 计算雅可比矩阵
          Eigen::Matrix<double, 1, 6> J_nq;
          J_nq.block<1, 3>(0, 0) = wld - plane.center;
          J_nq.block<1, 3>(0, 3) = -plane.normal;
          // 计算测量不确定度
          double sigma_l = J_nq * plane.plane_var * J_nq.transpose();
          sigma_l += plane.normal.transpose() * var_wld * plane.normal;

          // 如果点到平面的距离在3倍标准差内,认为匹配成功
          if (dis_to_plane < 3 * sqrt(sigma_l))
          {
            // 注释掉的代码是计算匹配概率
            // float prob = 1 / (sqrt(sigma_l)) * exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
            // if(prob > max_prob)
            {
              oc = this; // 记录匹配的节点
              sigma_d = sigma_l; // 记录不确定度
              // max_prob = prob;
              pla = &plane; // 记录匹配的平面
            }

            flag = 1; // 标记匹配成功
          }
        }
      }
    }
    else // 如果是非叶子节点
    {
      // 计算点所在的子节点编号
      int xyz[3] = {0, 0, 0};
      for (int k = 0; k < 3; k++)
        if (wld[k] > voxel_center[k])
          xyz[k] = 1;
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];

      // 注释掉的代码是遍历所有子节点进行匹配
      // for(int i=0; i<8; i++)
      // if(leaves[i] != nullptr)
      // {
      //   int flg = leaves[i]->match(wld, pla, max_prob, var_wld);
      //   if(i == leafnum)
      //     flag = flg;
      // }

      // 只在点所在的子节点中进行匹配
      if (leaves[leafnum] != nullptr)
        flag = leaves[leafnum]->match(wld, pla, max_prob, var_wld, sigma_d, oc);

      // 注释掉的代码是另一种遍历方式
      // for(int i=0; i<8; i++)
      //   if(leaves[i] != nullptr)
      //     leaves[i]->match(pv, pla, max_prob, var_wld);
    }

    return flag;
  }

  /**
   * 收集八叉树中所有子节点指针到列表中
   * 该函数用于遍历八叉树结构并将所有子节点添加到指定的列表中
   * 通常用于内存管理,如批量释放节点等操作
   * @param octos_release 用于存储收集到的节点指针列表
   */
  void tras_ptr(vector<OctoTree *> &octos_release)
  {
    // 只处理内部节点(非叶子节点)
    if (octo_state == 1)
    {
      // 遍历所有8个可能的子节点
      for (int i = 0; i < 8; i++)
        // 对非空子节点进行处理
        if (leaves[i] != nullptr)
        {
          // 将子节点指针添加到列表中
          octos_release.push_back(leaves[i]);
          // 递归地收集子节点的子节点
          leaves[i]->tras_ptr(octos_release);
        }
    }
  }

  // ~OctoTree()
  // {
  //   for(int i=0; i<8; i++)
  //   if(leaves[i] != nullptr)
  //   {
  //     delete leaves[i];
  //     leaves[i] = nullptr;
  //   }
  // }

  /**
   * 提取八叉树中的点云数据用于可视化调试
   * 该函数遍历八叉树结构,收集满足特定条件的点云并转换到指定的输出点云容器中
   * @param win_count 滑动窗口中的帧数
   * @param pl_fixd 输出参数,用于存储固定点云(注意:当前实现中此参数未被实际使用,相关代码被注释)
   * @param pl_wind 输出参数,用于存储滑动窗口中的动态点云
   * @param x_buf 包含位姿转换信息的缓冲区,用于将点从局部坐标系转换到全局坐标系
   */
  void tras_display(int win_count, pcl::PointCloud<PointType> &pl_fixd, pcl::PointCloud<PointType> &pl_wind, vector<IMUST> &x_buf)
  {
    // 处理叶子节点
    if (octo_state == 0)
    {
      // 计算点云协方差矩阵的特征值和特征向量
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
      Eigen::Matrix3d eig_vectors = saes.eigenvectors();
      Eigen::Vector3d eig_values = saes.eigenvalues();

      // 定义点云中的点
      PointType ap;
      // ap.intensity = ins; // 注释掉的强度设置

      // 如果当前节点包含有效平面
      if (plane.is_plane)
      {
        // 以下被注释的代码用于筛选特定条件的平面
        // if(pcr_add.N-pcr_fix.N < min_ba_point) return;
        // if(eig_value[0]/eig_value[1] > 0.1)
        //   return;

        // 以下被注释的代码用于处理固定点云
        // for(pointVar &pv: point_fix)
        // {
        //   Eigen::Vector3d pvec = pv.pnt;
        //   ap.x = pvec[0];
        //   ap.y = pvec[1];
        //   ap.z = pvec[2];
        //   ap.normal_x = sqrt(eig_values[0]);
        //   ap.normal_y = sqrt(eig_values[2] / eig_values[0]);
        //   ap.normal_z = pcr_add.N;
        //   ap.curvature = pcr_add.N - pcr_fix.N;
        //   pl_fixd.push_back(ap);
        // }

        // 遍历所有窗口中的帧
        for (int i = 0; i < win_count; i++)
          // 处理每一帧中的点
          for (pointVar &pv : sw->points[mp[i]])
          {
            // 将点从局部坐标系转换到全局坐标系
            Eigen::Vector3d pvec = x_buf[i].R * pv.pnt + x_buf[i].p;
            ap.x = pvec[0];
            ap.y = pvec[1];
            ap.z = pvec[2];
            // 以下被注释的代码用于设置点的法线和曲率等属性
            // ap.normal_x = sqrt(eig_values[0]);
            // ap.normal_y = sqrt(eig_values[2] / eig_values[0]);
            // ap.normal_z = pcr_add.N;
            // ap.curvature = pcr_add.N - pcr_fix.N;
            pl_wind.push_back(ap);
          }
      }
    }
    else // 处理非叶子节点
    {
      // 递归处理所有非空子节点
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
          leaves[i]->tras_display(win_count, pl_fixd, pl_wind, x_buf);
    }
  }

  /**
   * 判断给定三维点是否在当前体素内部
   * 该函数通过比较点的坐标与体素的边界来判断点是否位于体素内
   * 在点云处理、特征匹配和空间查询等操作中非常重要
   * @param wld 待检测的三维点坐标(以引用方式传入)
   * @return 如果点在体素内部返回true，否则返回false
   */
  bool inside(Eigen::Vector3d &wld)
  {
    // 计算体素半长度(quater_length是四分之一长度，所以乘以2得到半长度)
    double hl = quater_length * 2;
    // 检查点是否在体素的六个面所围成的立方体内部
    // 即检查点的每个坐标是否在体素中心正负半长度的范围内
    return (wld[0] >= voxel_center[0] - hl &&
            wld[0] <= voxel_center[0] + hl &&
            wld[1] >= voxel_center[1] - hl &&
            wld[1] <= voxel_center[1] + hl &&
            wld[2] >= voxel_center[2] - hl &&
            wld[2] <= voxel_center[2] + hl);
  }

  /**
   * 清理八叉树中的滑动窗口数据
   * 该函数递归地释放整个八叉树中所有节点关联的滑动窗口资源
   * 并将清空的滑动窗口对象收集到传入的列表中以便后续复用
   * @param sws 用于收集释放后的滑动窗口对象的列表
   */
  void clear_slwd(vector<SlideWindow *> &sws)
  {
    // 如果不是叶子节点，递归处理所有子节点
    if (octo_state != 0)
    {
      // 遍历所有可能的子节点
      for (int i = 0; i < 8; i++)
        // 对非空子节点递归调用清理函数
        if (leaves[i] != nullptr)
        {
          leaves[i]->clear_slwd(sws);
        }
    }

    // 处理当前节点的滑动窗口
    if (sw != nullptr)
    {
      // 清空滑动窗口中的数据
      sw->clear();
      // 将清空后的滑动窗口对象添加到回收列表
      sws.push_back(sw);
      // 断开当前节点与滑动窗口的关联
      sw = nullptr;
    }
  }
};

/**
 * 将点云数据分配到体素网格中并建立或更新八叉树结构
 * 该函数处理一帧点云数据，将每个点根据其空间位置分配到相应的体素中
 * 并在需要时创建新的八叉树节点或更新现有节点
 * 
 * @param feat_map 存储所有体素及其对应八叉树节点的哈希表
 * @param pvec 输入点云数据指针，包含点的位置和协方差信息
 * @param win_count 当前滑动窗口的索引或帧计数
 * @param feat_tem_map 临时哈希表，用于存储当前帧处理过的体素
 * @param wdsize 滑动窗口的大小
 * @param pwld 世界坐标系下的点云位置向量
 * @param sws 可重用的滑动窗口对象列表
 */
void cut_voxel(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, PVecPtr pvec, int win_count, unordered_map<VOXEL_LOC, OctoTree *> &feat_tem_map, int wdsize, PLV(3) & pwld, vector<SlideWindow *> &sws)
{
  // 获取点云大小
  int plsize = pvec->size();
  // 遍历所有点
  for (int i = 0; i < plsize; i++)
  {
    // 获取当前点的引用及其世界坐标
    pointVar &pv = (*pvec)[i];
    Eigen::Vector3d &pw = pwld[i];
    // 计算点所在的体素索引
    float loc[3];
    for (int j = 0; j < 3; j++)
    {
      // 将世界坐标除以体素大小得到体素索引
      loc[j] = pw[j] / voxel_size;
      // 负坐标的特殊处理，确保正确映射到整数索引
      if (loc[j] < 0)
        loc[j] -= 1;
    }

    // 创建体素位置标识符
    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    // 在现有体素表中查找当前位置
    auto iter = feat_map.find(position);
    // 如果该体素已存在
    if (iter != feat_map.end())
    {
      // 将点分配到现有八叉树节点
      iter->second->allocate(win_count, pv, pw, sws);
      // 标记该节点包含有效数据
      iter->second->isexist = true;
      // 将该体素添加到临时表中(如果尚未添加)
      if (feat_tem_map.find(position) == feat_map.end())
        feat_tem_map[position] = iter->second;
    }
    else // 如果该位置没有体素
    {
      // 创建新的八叉树节点(初始层级为0)
      OctoTree *ot = new OctoTree(0, wdsize);
      // 分配点到新节点
      ot->allocate(win_count, pv, pw, sws);
      // 设置体素中心坐标(加0.5是为了获取体素中心而非边缘)
      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      // 设置体素四分之一长度(用于子节点划分)
      ot->quater_length = voxel_size / 4.0;
      // 将新节点添加到体素表和临时表中
      feat_map[position] = ot;
      feat_tem_map[position] = ot;
    }
  }
}

/**
 * 多线程版本的点云体素化与八叉树构建函数
 * 该函数通过多线程并行处理方式，将点云数据分配到空间体素中
 * 相比单线程版本，提高了大规模点云处理的效率
 * 
 * @param feat_map 存储所有体素及其对应八叉树节点的哈希表
 * @param pvec 输入点云数据指针，包含点的位置和协方差信息
 * @param win_count 当前滑动窗口的索引或帧计数
 * @param feat_tem_map 临时哈希表，用于存储当前帧处理过的体素
 * @param wdsize 滑动窗口的大小
 * @param pwld 世界坐标系下的点云位置向量
 * @param sws 多线程使用的滑动窗口对象二维数组，每个线程使用一组滑动窗口
 */
void cut_voxel_multi(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, PVecPtr pvec, int win_count, unordered_map<VOXEL_LOC, OctoTree *> &feat_tem_map, int wdsize, PLV(3) & pwld, vector<vector<SlideWindow *>> &sws)
{
  // 创建映射表，记录每个八叉树节点对应的点云索引
  unordered_map<OctoTree *, vector<int>> map_pvec;
  // 获取点云大小
  int plsize = pvec->size();
  // 第一阶段：遍历所有点，确定每个点所属的体素
  for (int i = 0; i < plsize; i++)
  {
    // 获取当前点的引用及其世界坐标
    pointVar &pv = (*pvec)[i];
    Eigen::Vector3d &pw = pwld[i];
    // 计算点所在的体素索引
    float loc[3];
    for (int j = 0; j < 3; j++)
    {
      // 注释掉的代码可能是旧版本实现
      // loc[j] = pv.world[j] / voxel_size;
      // 将世界坐标除以体素大小得到体素索引
      loc[j] = pw[j] / voxel_size;
      // 负坐标的特殊处理，确保正确映射到整数索引
      if (loc[j] < 0)
        loc[j] -= 1;
    }

    // 创建体素位置标识符
    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    // 在现有体素表中查找当前位置
    auto iter = feat_map.find(position);
    OctoTree *ot = nullptr;
    // 如果该体素已存在
    if (iter != feat_map.end())
    {
      // 标记该节点包含有效数据
      iter->second->isexist = true;
      // 将该体素添加到临时表中(如果尚未添加)
      if (feat_tem_map.find(position) == feat_map.end())
        feat_tem_map[position] = iter->second;
      // 获取体素对应的八叉树节点
      ot = iter->second;
    }
    else // 如果该位置没有体素
    {
      // 创建新的八叉树节点(初始层级为0)
      ot = new OctoTree(0, wdsize);
      // 设置体素中心坐标(加0.5是为了获取体素中心而非边缘)
      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      // 设置体素四分之一长度(用于子节点划分)
      ot->quater_length = voxel_size / 4.0;
      // 将新节点添加到体素表和临时表中
      feat_map[position] = ot;
      feat_tem_map[position] = ot;
    }

    // 记录点云索引到对应的八叉树节点
    map_pvec[ot].push_back(i);
  }

  // 注释掉的单线程实现代码
  // for(auto iter=map_pvec.begin(); iter!=map_pvec.end(); iter++)
  // {
  //   for(int i: iter->second)
  //   {
  //     iter->first->allocate(win_count, (*pvec)[i], pwld[i], sws);
  //   }
  // }

  // 将哈希表转换为向量以便并行处理
  vector<pair<OctoTree *const, vector<int>> *> octs;
  octs.reserve(map_pvec.size());
  for (auto iter = map_pvec.begin(); iter != map_pvec.end(); iter++)
    octs.push_back(&(*iter));

  // 线程数等于滑动窗口组的数量
  int thd_num = sws.size();
  // 获取总的体素数量
  int g_size = octs.size();
  // 如果体素数量少于线程数，直接返回
  if (g_size < thd_num)
    return;
  // 创建线程数组
  vector<thread *> mthreads(thd_num);
  // 计算每个线程需要处理的体素数量
  double part = 1.0 * g_size / thd_num;

  // 重新分配滑动窗口资源，确保每个线程有足够的滑动窗口
  int swsize = sws[0].size() / thd_num;
  for (int i = 1; i < thd_num; i++)
  {
    // 将部分滑动窗口从第一组移动到其他组
    sws[i].insert(sws[i].end(), sws[0].end() - swsize, sws[0].end());
    sws[0].erase(sws[0].end() - swsize, sws[0].end());
  }

  // 创建并启动工作线程(除了主线程外)
  for (int i = 1; i < thd_num; i++)
  {
    mthreads[i] = new thread(
        // 线程工作函数：处理分配给该线程的体素和点云
        [&](int head, int tail, vector<SlideWindow *> &sw)
        {
          for (int j = head; j < tail; j++)
          {
            // 遍历该体素中的所有点索引
            for (int k : octs[j]->second)
              // 将点分配到八叉树节点
              octs[j]->first->allocate(win_count, (*pvec)[k], pwld[k], sw);
          }
        },
        // 线程处理的体素范围和使用的滑动窗口组
        part * i, part * (i + 1), ref(sws[i]));
  }

  // 等待所有线程完成并释放资源
  for (int i = 0; i < thd_num; i++)
  {
    if (i == 0)
    {
      // 主线程处理第一部分体素
      for (int j = 0; j < int(part); j++)
        for (int k : octs[j]->second)
          octs[j]->first->allocate(win_count, (*pvec)[k], pwld[k], sws[0]);
    }
    else
    {
      // 等待其他线程完成并释放资源
      mthreads[i]->join();
      delete mthreads[i];
    }
  }
}

/**
 * 处理固定点云数据的体素化函数
 * 该函数将固定点云分配到八叉树结构中，用于地图构建和维护
 * 与处理动态点云的cut_voxel函数不同，此函数专门处理不随位姿优化变化的固定点
 * 
 * @param feat_map 存储所有体素及其对应八叉树节点的哈希表
 * @param pvec 输入固定点云数据，包含点的位置和协方差信息
 * @param wdsize 滑动窗口的大小
 * @param jour 时间戳或序号，用于记录点云添加的时间
 */
void cut_voxel(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, PVec &pvec, int wdsize, double jour)
{
  // 遍历所有固定点
  for (pointVar &pv : pvec)
  {
    // 计算点所在的体素索引
    float loc[3];
    for (int j = 0; j < 3; j++)
    {
      // 将点坐标除以体素大小得到体素索引
      loc[j] = pv.pnt[j] / voxel_size;
      // 负坐标的特殊处理，确保正确映射到整数索引
      if (loc[j] < 0)
        loc[j] -= 1;
    }

    // 创建体素位置标识符
    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    // 在现有体素表中查找当前位置
    auto iter = feat_map.find(position);
    // 如果该体素已存在
    if (iter != feat_map.end())
    {
      // 将固定点添加到现有八叉树节点
      iter->second->allocate_fix(pv);
    }
    else // 如果该位置没有体素
    {
      // 创建新的八叉树节点(初始层级为0)
      OctoTree *ot = new OctoTree(0, wdsize);
      // 添加固定点到新节点(不带协方差)
      ot->push_fix_novar(pv);
      // 设置体素中心坐标(加0.5是为了获取体素中心而非边缘)
      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      // 设置体素四分之一长度(用于子节点划分)
      ot->quater_length = voxel_size / 4.0;
      // 记录时间戳或序号
      ot->jour = jour;
      // 将新节点添加到体素表中
      feat_map[position] = ot;
    }
  }
}

/**
 * 在体素地图中进行点与平面的匹配
 * 该函数在八叉树体素地图中查找给定三维点所在的体素，并尝试将其与该体素中的平面特征进行匹配
 * 通常用于点云配准、特征关联或数据关联等任务
 * 
 * @param feat_map 体素地图，一个哈希表，键为体素位置，值为八叉树节点指针
 * @param wld 待匹配的三维点坐标(以引用方式传入)
 * @param pla 输出参数，指向匹配到的平面的指针(如果匹配成功)
 * @param var_wld 输出参数，匹配点的协方差矩阵
 * @param sigma_d 输出参数，点到平面的标准差或不确定性度量
 * @param oc 输出参数，匹配到的八叉树节点指针
 * @return 匹配结果标志：0表示未匹配，非0表示匹配成功
 */
int match(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, Eigen::Vector3d &wld, Plane *&pla, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree *&oc)
{
  // 初始化匹配标志为0(未匹配)
  int flag = 0;

  // 计算点所在的体素位置
  float loc[3];
  for (int j = 0; j < 3; j++)
  {
    // 将世界坐标除以体素大小得到体素坐标
    loc[j] = wld[j] / voxel_size;
    // 负坐标需要向下取整，以确保索引的正确性
    if (loc[j] < 0)
      loc[j] -= 1;
  }
  
  // 构造体素位置索引对象
  VOXEL_LOC position(loc[0], loc[1], loc[2]);
  
  // 在体素地图中查找对应位置的节点
  auto iter = feat_map.find(position);
  // 如果找到了对应的体素节点
  if (iter != feat_map.end())
  {
    // 初始化最大匹配概率
    double max_prob = 0;
    // 调用找到的体素节点的match方法进行点与平面的详细匹配
    flag = iter->second->match(wld, pla, max_prob, var_wld, sigma_d, oc);
    
    // 调试检查：如果匹配成功但平面指针为空，输出错误信息
    if (flag && pla == nullptr)
    {
      printf("pla null max_prob: %lf %ld %ld %ld\n", max_prob, iter->first.x, iter->first.y, iter->first.z);
    }
  }

  // 返回匹配结果
  return flag;
}

#endif
