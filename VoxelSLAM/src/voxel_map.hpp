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

struct pointVar
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Vector3d pnt;
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
  SlideWindow *sw = nullptr;
  PointCluster pcr_add;
  Eigen::Matrix<double, 9, 9> cov_add;

  PointCluster pcr_fix;
  PVec point_fix;

  int layer, octo_state, wdsize;
  OctoTree *leaves[8];
  double voxel_center[3];
  double jour = 0;
  float quater_length;

  Plane plane;
  bool isexist = false;

  Eigen::Vector3d eig_value;
  Eigen::Matrix3d eig_vector;

  int last_num = 0, opt_state = -1;
  mutex mVox;

  OctoTree(int _l, int _w) : layer(_l), wdsize(_w), octo_state(0)
  {
    for (int i = 0; i < 8; i++)
      leaves[i] = nullptr;
    cov_add.setZero();

    // ins = 255.0*rand()/(RAND_MAX + 1.0f);
  }

  inline void push(int ord, const pointVar &pv, const Eigen::Vector3d &pw, vector<SlideWindow *> &sws)
  {
    mVox.lock();
    if (sw == nullptr)
    {
      if (sws.size() != 0)
      {
        sw = sws.back();
        sws.pop_back();
        sw->resize(wdsize);
      }
      else
        sw = new SlideWindow(wdsize);
    }
    if (!isexist)
      isexist = true;

    int mord = mp[ord];
    if (layer < max_layer)
      sw->points[mord].push_back(pv);
    sw->pcrs_local[mord].push(pv.pnt);
    pcr_add.push(pw);
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pw);
    cov_add += Bi;
    mVox.unlock();
  }

  inline void push_fix(pointVar &pv)
  {
    if (layer < max_layer)
      point_fix.push_back(pv);
    pcr_fix.push(pv.pnt);
    pcr_add.push(pv.pnt);
    Eigen::Matrix<double, 9, 9> Bi;
    Bf_var(pv, Bi, pv.pnt);
    cov_add += Bi;
  }

  inline void push_fix_novar(pointVar &pv)
  {
    if (layer < max_layer)
      point_fix.push_back(pv);
    pcr_fix.push(pv.pnt);
    pcr_add.push(pv.pnt);
  }

  inline bool plane_judge(Eigen::Vector3d &eig_values)
  {
    // return (eig_values[0] < min_eigen_value);
    return (eig_values[0] < min_eigen_value && (eig_values[0] / eig_values[2]) < plane_eigen_value_thre[layer]);
  }

  void allocate(int ord, const pointVar &pv, const Eigen::Vector3d &pw, vector<SlideWindow *> &sws)
  {
    if (octo_state == 0)
    {
      push(ord, pv, pw, sws);
    }
    else
    {
      int xyz[3] = {0, 0, 0};
      for (int k = 0; k < 3; k++)
        if (pw[k] > voxel_center[k])
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

  void fix_divide(vector<SlideWindow *> &sws)
  {
    for (pointVar &pv : point_fix)
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

      leaves[leafnum]->push_fix(pv);
    }
  }

  void subdivide(int si, IMUST &xx, vector<SlideWindow *> &sws)
  {
    for (pointVar &pv : sw->points[mp[si]])
    {
      Eigen::Vector3d pw = xx.R * pv.pnt + xx.p;
      int xyz[3] = {0, 0, 0};
      for (int k = 0; k < 3; k++)
        if (pw[k] > voxel_center[k])
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

      leaves[leafnum]->push(si, pv, pw, sws);
    }
  }

  void plane_update()
  {
    plane.center = pcr_add.v / pcr_add.N;
    int l = 0;
    Eigen::Vector3d u[3] = {eig_vector.col(0), eig_vector.col(1), eig_vector.col(2)};
    double nv = 1.0 / pcr_add.N;

    Eigen::Matrix<double, 3, 9> u_c;
    u_c.setZero();
    for (int k = 0; k < 3; k++)
      if (k != l)
      {
        Eigen::Matrix3d ukl = u[k] * u[l].transpose();
        Eigen::Matrix<double, 1, 9> fkl;
        fkl.head(6) << ukl(0, 0), ukl(1, 0) + ukl(0, 1), ukl(2, 0) + ukl(0, 2),
            ukl(1, 1), ukl(1, 2) + ukl(2, 1), ukl(2, 2);
        fkl.tail(3) = -(u[k].dot(plane.center) * u[l] + u[l].dot(plane.center) * u[k]);

        u_c += nv / (eig_value[l] - eig_value[k]) * u[k] * fkl;
      }

    Eigen::Matrix<double, 3, 9> Jc = u_c * cov_add;
    plane.plane_var.block<3, 3>(0, 0) = Jc * u_c.transpose();
    Eigen::Matrix3d Jc_N = nv * Jc.block<3, 3>(0, 6);
    plane.plane_var.block<3, 3>(0, 3) = Jc_N;
    plane.plane_var.block<3, 3>(3, 0) = Jc_N.transpose();
    plane.plane_var.block<3, 3>(3, 3) = nv * nv * cov_add.block<3, 3>(6, 6);
    plane.normal = u[0];
    plane.radius = eig_value[2];
  }

  void recut(int win_count, vector<IMUST> &x_buf, vector<SlideWindow *> &sws)
  {
    if (octo_state == 0)
    {
      if (layer >= 0)
      {
        opt_state = -1;
        if (pcr_add.N <= min_point[layer])
        {
          plane.is_plane = false;
          return;
        }
        if (!isexist || sw == nullptr)
          return;

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
        eig_value = saes.eigenvalues();
        eig_vector = saes.eigenvectors();
        plane.is_plane = plane_judge(eig_value);

        if (plane.is_plane)
        {
          return;
        }
        else if (layer >= max_layer)
          return;
      }

      if (pcr_fix.N != 0)
      {
        fix_divide(sws);
        // point_fix.clear();
        PVec().swap(point_fix);
      }

      for (int i = 0; i < win_count; i++)
        subdivide(i, x_buf[i], sws);

      sw->clear();
      sws.push_back(sw);
      sw = nullptr;
      octo_state = 1;
    }

    for (int i = 0; i < 8; i++)
      if (leaves[i] != nullptr)
        leaves[i]->recut(win_count, x_buf, sws);
  }

  void margi(int win_count, int mgsize, vector<IMUST> &x_buf, const LidarFactor &vox_opt)
  {
    if (octo_state == 0 && layer >= 0)
    {
      if (!isexist || sw == nullptr)
        return;
      mVox.lock();
      vector<PointCluster> pcrs_world(wdsize);
      // pcr_add = pcr_fix;
      // for(int i=0; i<win_count; i++)
      // if(sw->pcrs_local[mp[i]].N != 0)
      // {
      //   pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
      //   pcr_add += pcrs_world[i];
      // }

      if (opt_state >= int(vox_opt.pcr_adds.size()))
      {
        printf("Error: opt_state: %d %zu\n", opt_state, vox_opt.pcr_adds.size());
        exit(0);
      }

      if (opt_state >= 0)
      {
        pcr_add = vox_opt.pcr_adds[opt_state];
        eig_value = vox_opt.eig_values[opt_state];
        eig_vector = vox_opt.eig_vectors[opt_state];
        opt_state = -1;

        for (int i = 0; i < mgsize; i++)
          if (sw->pcrs_local[mp[i]].N != 0)
          {
            pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
          }
      }
      else
      {
        pcr_add = pcr_fix;
        for (int i = 0; i < win_count; i++)
          if (sw->pcrs_local[mp[i]].N != 0)
          {
            pcrs_world[i].transform(sw->pcrs_local[mp[i]], x_buf[i]);
            pcr_add += pcrs_world[i];
          }

        if (plane.is_plane)
        {
          Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
          eig_value = saes.eigenvalues();
          eig_vector = saes.eigenvectors();
        }
      }

      if (pcr_fix.N < max_points && plane.is_plane)
        if (pcr_add.N - last_num >= 5 || last_num <= 10)
        {
          plane_update();
          last_num = pcr_add.N;
        }

      if (pcr_fix.N < max_points)
      {
        for (int i = 0; i < mgsize; i++)
          if (pcrs_world[i].N != 0)
          {
            pcr_fix += pcrs_world[i];
            for (pointVar pv : sw->points[mp[i]])
            {
              pv.pnt = x_buf[i].R * pv.pnt + x_buf[i].p;
              point_fix.push_back(pv);
            }
          }
      }
      else
      {
        for (int i = 0; i < mgsize; i++)
          if (pcrs_world[i].N != 0)
            pcr_add -= pcrs_world[i];

        if (point_fix.size() != 0)
          PVec().swap(point_fix);
      }

      for (int i = 0; i < mgsize; i++)
        if (sw->pcrs_local[mp[i]].N != 0)
        {
          sw->pcrs_local[mp[i]].clear();
          sw->points[mp[i]].clear();
        }

      if (pcr_fix.N >= pcr_add.N)
        isexist = false;
      else
        isexist = true;

      mVox.unlock();
    }
    else
    {
      isexist = false;
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
        {
          leaves[i]->margi(win_count, mgsize, x_buf, vox_opt);
          isexist = isexist || leaves[i]->isexist;
        }
    }
  }

  // Extract the LiDAR factor
  void tras_opt(LidarFactor &vox_opt)
  {
    if (octo_state == 0)
    {
      if (layer >= 0 && isexist && plane.is_plane && sw != nullptr)
      {
        if (eig_value[0] / eig_value[1] > 0.12)
          return;

        double coe = 1;
        vector<PointCluster> pcrs(wdsize);
        for (int i = 0; i < wdsize; i++)
          pcrs[i] = sw->pcrs_local[mp[i]];
        opt_state = vox_opt.plvec_voxels.size();
        vox_opt.push_voxel(pcrs, pcr_fix, coe, eig_value, eig_vector, pcr_add);
      }
    }
    else
    {
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

  void tras_ptr(vector<OctoTree *> &octos_release)
  {
    if (octo_state == 1)
    {
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
        {
          octos_release.push_back(leaves[i]);
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

  // Extract the point cloud map for debug
  void tras_display(int win_count, pcl::PointCloud<PointType> &pl_fixd, pcl::PointCloud<PointType> &pl_wind, vector<IMUST> &x_buf)
  {
    if (octo_state == 0)
    {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(pcr_add.cov());
      Eigen::Matrix3d eig_vectors = saes.eigenvectors();
      Eigen::Vector3d eig_values = saes.eigenvalues();

      PointType ap;
      // ap.intensity = ins;

      if (plane.is_plane)
      {
        // if(pcr_add.N-pcr_fix.N < min_ba_point) return;
        // if(eig_value[0]/eig_value[1] > 0.1)
        //   return;

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

        for (int i = 0; i < win_count; i++)
          for (pointVar &pv : sw->points[mp[i]])
          {
            Eigen::Vector3d pvec = x_buf[i].R * pv.pnt + x_buf[i].p;
            ap.x = pvec[0];
            ap.y = pvec[1];
            ap.z = pvec[2];
            // ap.normal_x = sqrt(eig_values[0]);
            // ap.normal_y = sqrt(eig_values[2] / eig_values[0]);
            // ap.normal_z = pcr_add.N;
            // ap.curvature = pcr_add.N - pcr_fix.N;
            pl_wind.push_back(ap);
          }
      }
    }
    else
    {
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
          leaves[i]->tras_display(win_count, pl_fixd, pl_wind, x_buf);
    }
  }

  bool inside(Eigen::Vector3d &wld)
  {
    double hl = quater_length * 2;
    return (wld[0] >= voxel_center[0] - hl &&
            wld[0] <= voxel_center[0] + hl &&
            wld[1] >= voxel_center[1] - hl &&
            wld[1] <= voxel_center[1] + hl &&
            wld[2] >= voxel_center[2] - hl &&
            wld[2] <= voxel_center[2] + hl);
  }

  void clear_slwd(vector<SlideWindow *> &sws)
  {
    if (octo_state != 0)
    {
      for (int i = 0; i < 8; i++)
        if (leaves[i] != nullptr)
        {
          leaves[i]->clear_slwd(sws);
        }
    }

    if (sw != nullptr)
    {
      sw->clear();
      sws.push_back(sw);
      sw = nullptr;
    }
  }
};

// feat_map: 特征体素地图，是一个哈希表，键为体素位置(VOXEL_LOC)，值为八叉树节点指针
// pvec: 点云向量指针，存储了当前帧的所有点及其属性
// win_count: 当前点云在滑动窗口中的索引
// feat_tem_map: 临时体素地图，存储当前帧涉及的体素
// wdsize: 滑动窗口大小
// pwld: 世界坐标系下的点位置向量
// sws: 滑动窗口数据结构，用于管理点与位姿的关联
void cut_voxel(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, PVecPtr pvec, int win_count, unordered_map<VOXEL_LOC, OctoTree *> &feat_tem_map, int wdsize, PLV(3) & pwld, vector<SlideWindow *> &sws)
{
  int plsize = pvec->size(); // 获取点云中点的数量
  for (int i = 0; i < plsize; i++) // 遍历每个点
  {
    pointVar &pv = (*pvec)[i]; // 获取当前点的引用（包含点的位置和协方差）
    Eigen::Vector3d &pw = pwld[i]; // 获取对应的世界坐标点引用

    float loc[3]; // 用于存储体素坐标
    for (int j = 0; j < 3; j++) // 计算点所在的体素坐标
    {
      loc[j] = pw[j] / voxel_size;
      if (loc[j] < 0) // 处理负坐标的特殊情况
        loc[j] -= 1;
    }

    VOXEL_LOC position(loc[0], loc[1], loc[2]); // 创建体素位置对象
    auto iter = feat_map.find(position); // 在体素地图中查找该位置
    if (iter != feat_map.end()) // 如果该体素已存在
    {
      iter->second->allocate(win_count, pv, pw, sws); // 将点分配到现有八叉树
      iter->second->isexist = true; // 标记该体素节点存在点
      if (feat_tem_map.find(position) == feat_map.end()) // 如果临时地图中不存在该体素
        feat_tem_map[position] = iter->second; // 将其添加到临时地图
    }
    else // 如果体素不存在
    {
      OctoTree *ot = new OctoTree(0, wdsize); // 创建新的八叉树节点
      ot->allocate(win_count, pv, pw, sws); // 分配点到该节点
      // 设置体素中心坐标（体素索引坐标 + 0.5 然后乘以体素大小）
      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0; // 设置四分之一边长（八叉树细分时使用）
      feat_map[position] = ot; // 将新节点添加到主体素地图
      feat_tem_map[position] = ot; // 同时添加到临时地图
    }
  }
}

// Cut the current scan into corresponding voxel in multi thread
void cut_voxel_multi(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, PVecPtr pvec, int win_count, unordered_map<VOXEL_LOC, OctoTree *> &feat_tem_map, int wdsize, PLV(3) & pwld, vector<vector<SlideWindow *>> &sws)
{
  unordered_map<OctoTree *, vector<int>> map_pvec;
  int plsize = pvec->size();
  for (int i = 0; i < plsize; i++)
  {
    pointVar &pv = (*pvec)[i];
    Eigen::Vector3d &pw = pwld[i];
    float loc[3];
    for (int j = 0; j < 3; j++)
    {
      // loc[j] = pv.world[j] / voxel_size;
      loc[j] = pw[j] / voxel_size;
      if (loc[j] < 0)
        loc[j] -= 1;
    }

    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    auto iter = feat_map.find(position);
    OctoTree *ot = nullptr;
    if (iter != feat_map.end())
    {
      iter->second->isexist = true;
      if (feat_tem_map.find(position) == feat_map.end())
        feat_tem_map[position] = iter->second;
      ot = iter->second;
    }
    else
    {
      ot = new OctoTree(0, wdsize);
      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      feat_map[position] = ot;
      feat_tem_map[position] = ot;
    }

    map_pvec[ot].push_back(i);
  }

  // for(auto iter=map_pvec.begin(); iter!=map_pvec.end(); iter++)
  // {
  //   for(int i: iter->second)
  //   {
  //     iter->first->allocate(win_count, (*pvec)[i], pwld[i], sws);
  //   }
  // }

  vector<pair<OctoTree *const, vector<int>> *> octs;
  octs.reserve(map_pvec.size());
  for (auto iter = map_pvec.begin(); iter != map_pvec.end(); iter++)
    octs.push_back(&(*iter));

  int thd_num = sws.size();
  int g_size = octs.size();
  if (g_size < thd_num)
    return;
  vector<thread *> mthreads(thd_num);
  double part = 1.0 * g_size / thd_num;

  int swsize = sws[0].size() / thd_num;
  for (int i = 1; i < thd_num; i++)
  {
    sws[i].insert(sws[i].end(), sws[0].end() - swsize, sws[0].end());
    sws[0].erase(sws[0].end() - swsize, sws[0].end());
  }

  for (int i = 1; i < thd_num; i++)
  {
    mthreads[i] = new thread(
        [&](int head, int tail, vector<SlideWindow *> &sw)
        {
          for (int j = head; j < tail; j++)
          {
            for (int k : octs[j]->second)
              octs[j]->first->allocate(win_count, (*pvec)[k], pwld[k], sw);
          }
        },
        part * i, part * (i + 1), ref(sws[i]));
  }

  for (int i = 0; i < thd_num; i++)
  {
    if (i == 0)
    {
      for (int j = 0; j < int(part); j++)
        for (int k : octs[j]->second)
          octs[j]->first->allocate(win_count, (*pvec)[k], pwld[k], sws[0]);
    }
    else
    {
      mthreads[i]->join();
      delete mthreads[i];
    }
  }
}

void cut_voxel(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, PVec &pvec, int wdsize, double jour)
{
  for (pointVar &pv : pvec)
  {
    float loc[3];
    for (int j = 0; j < 3; j++)
    {
      loc[j] = pv.pnt[j] / voxel_size;
      if (loc[j] < 0)
        loc[j] -= 1;
    }

    VOXEL_LOC position(loc[0], loc[1], loc[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end())
    {
      iter->second->allocate_fix(pv);
    }
    else
    {
      OctoTree *ot = new OctoTree(0, wdsize);
      ot->push_fix_novar(pv);
      ot->voxel_center[0] = (0.5 + position.x) * voxel_size;
      ot->voxel_center[1] = (0.5 + position.y) * voxel_size;
      ot->voxel_center[2] = (0.5 + position.z) * voxel_size;
      ot->quater_length = voxel_size / 4.0;
      ot->jour = jour;
      feat_map[position] = ot;
    }
  }
}

// Match the point with the plane in the voxel map
int match(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, Eigen::Vector3d &wld, Plane *&pla, Eigen::Matrix3d &var_wld, double &sigma_d, OctoTree *&oc)
{
  int flag = 0;

  // 计算点所在的体素位置
  float loc[3];
  for (int j = 0; j < 3; j++)
  {
    loc[j] = wld[j] / voxel_size; // 将世界坐标除以体素大小得到体素坐标
    if (loc[j] < 0)
      loc[j] -= 1; // 负坐标需要向下取整
  }
  
  // 构造体素位置索引
  VOXEL_LOC position(loc[0], loc[1], loc[2]);
  
  // 在体素地图中查找对应位置的节点
  auto iter = feat_map.find(position);
  if (iter != feat_map.end())
  {
    double max_prob = 0;
    // 在找到的体素节点中进行点与平面的匹配
    flag = iter->second->match(wld, pla, max_prob, var_wld, sigma_d, oc);
    
    // 如果匹配成功但平面指针为空,输出错误信息
    if (flag && pla == nullptr)
    {
      printf("pla null max_prob: %lf %ld %ld %ld\n", max_prob, iter->first.x, iter->first.y, iter->first.z);
    }
  }

  return flag;
}

#endif
