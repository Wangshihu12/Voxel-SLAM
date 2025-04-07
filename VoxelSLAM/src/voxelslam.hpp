#pragma once

#include "tools.hpp"
#include "ekf_imu.hpp"
#include "voxel_map.hpp"
#include "feature_point.hpp"
#include "loop_refine.hpp"
#include <mutex>
#include <Eigen/Eigenvalues>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/MarkerArray.h>
#include <malloc.h>
#include <geometry_msgs/PoseArray.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <malloc.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include "BTC.h"

using namespace std;

ros::Publisher pub_scan, pub_cmap, pub_init, pub_pmap;
ros::Publisher pub_test, pub_prev_path, pub_curr_path;
ros::Subscriber sub_imu, sub_pcl;

template <typename T>
void pub_pl_func(T &pl, ros::Publisher &pub)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "camera_init";
  output.header.stamp = ros::Time::now();
  pub.publish(output);
}

mutex mBuf;
Features feat;
deque<sensor_msgs::Imu::Ptr> imu_buf;
deque<pcl::PointCloud<PointType>::Ptr> pcl_buf;
deque<double> time_buf;

double imu_last_time = -1;
int point_notime = 0;
double last_pcl_time = -1;

void imu_handler(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  static int flag = 1;
  if(flag)
  {
    flag = 0;
    printf("Time0: %lf\n", msg_in->header.stamp.toSec());
  }

  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  // For Hilti 2022 exp03
  // double t0 = 1646320760 + 255.5;
  // double t1 = 1646320760 + 256.2;
  // double tc = msg->header.stamp.toSec();
  // if(tc > t0 && tc < t1)
  //   msg->linear_acceleration.z = -9.7;

  mBuf.lock();
  imu_last_time = msg->header.stamp.toSec();
  imu_buf.push_back(msg);
  mBuf.unlock();
}

template<class T>
void pcl_handler(T &msg)
{
  pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
  double t0 = feat.process(msg, *pl_ptr);

  if(pl_ptr->empty())
  {
    PointType ap; 
    ap.x = 0; ap.y = 0; ap.z = 0; 
    ap.intensity = 0; ap.curvature = 0;
    pl_ptr->push_back(ap);
    ap.curvature = 0.09;
    pl_ptr->push_back(ap);
  }

  sort(pl_ptr->begin(), pl_ptr->end(), [](PointType &x, PointType &y)
  {
    return x.curvature < y.curvature;
  });
  while(pl_ptr->back().curvature > 0.11)
    pl_ptr->points.pop_back();

  mBuf.lock();
  time_buf.push_back(t0);
  pcl_buf.push_back(pl_ptr);
  mBuf.unlock();
}

/**
 * @brief 同步点云和IMU数据包
 * @param pl_ptr 输出的点云数据指针
 * @param imus 输出的IMU数据队列
 * @param p_imu IMU EKF处理器
 * @return 如果成功同步返回true，否则返回false
 */
bool sync_packages(pcl::PointCloud<PointType>::Ptr &pl_ptr, deque<sensor_msgs::Imu::Ptr> &imus, IMUEKF &p_imu)
{
  static bool pl_ready = false;  // 标记点云数据是否准备好

  // 如果点云数据未准备好，尝试从缓冲区获取新的点云数据
  if(!pl_ready)
  {
    if(pcl_buf.empty()) return false;  // 如果点云缓冲区为空，返回false

    // 从缓冲区获取点云数据
    mBuf.lock();
    pl_ptr = pcl_buf.front();  // 获取最早的点云数据
    p_imu.pcl_beg_time = time_buf.front();  // 记录点云开始时间
    pcl_buf.pop_front(); time_buf.pop_front();  // 从缓冲区移除已处理的数据
    mBuf.unlock();

    // 计算点云结束时间（开始时间 + 最后一个点的曲率值）
    p_imu.pcl_end_time = p_imu.pcl_beg_time + pl_ptr->back().curvature;

    // 处理无时间戳的点云数据
    if(point_notime)
    {
      if(last_pcl_time < 0)  // 如果是第一个点云
      {
        last_pcl_time = p_imu.pcl_beg_time;
        return false;
      }

      // 更新时间戳
      p_imu.pcl_end_time = p_imu.pcl_beg_time;
      p_imu.pcl_beg_time = last_pcl_time;
      last_pcl_time = p_imu.pcl_end_time;
    }

    pl_ready = true;  // 标记点云数据已准备好
  }

  // 检查是否满足同步条件
  if(!pl_ready || imu_last_time <= p_imu.pcl_end_time) return false;

  // 从IMU缓冲区获取对应时间段的IMU数据
  mBuf.lock();
  double imu_time = imu_buf.front()->header.stamp.toSec();
  while((!imu_buf.empty()) && (imu_time < p_imu.pcl_end_time)) 
  {
    imu_time = imu_buf.front()->header.stamp.toSec();
    if(imu_time > p_imu.pcl_end_time) break;
    imus.push_back(imu_buf.front());  // 将IMU数据添加到输出队列
    imu_buf.pop_front();  // 从缓冲区移除已处理的IMU数据
  }
  mBuf.unlock();

  // 检查IMU缓冲区是否为空
  if(imu_buf.empty())
  {
    printf("imu buf empty\n"); exit(0);
  }

  pl_ready = false;  // 重置点云准备状态

  // 如果收集到足够多的IMU数据（>4个），返回true
  if(imus.size() > 4)
    return true;
  else
    return false;
}

double dept_err, beam_err;
void calcBodyVar(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &var) 
{
  if (pb[2] == 0)
    pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  var = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
};

/**
 * @brief 初始化点云中每个点的方差矩阵
 * @param ext 当前帧的IMU状态（包含旋转矩阵R和平移向量p）
 * @param pl_cur 当前帧的点云数据
 * @param pptr 存储点云方差信息的指针
 * @param dept_err 深度测量误差
 * @param beam_err 光束角度误差
 */
void var_init(IMUST &ext, pcl::PointCloud<PointType> &pl_cur, PVecPtr pptr, double dept_err, double beam_err)
{
  int plsize = pl_cur.size();  // 获取点云中点的数量
  pptr->clear();  // 清空方差容器
  pptr->resize(plsize);  // 重新分配空间
  
  // 遍历点云中的每个点
  for(int i=0; i<plsize; i++)
  {
    PointType &ap = pl_cur[i];  // 获取当前点
    pointVar &pv = pptr->at(i);  // 获取对应的方差结构体
    
    // 将点坐标赋值给方差结构体
    pv.pnt << ap.x, ap.y, ap.z;
    
    // 计算点在体坐标系下的方差矩阵
    calcBodyVar(pv.pnt, dept_err, beam_err, pv.var);
    
    // 将点转换到世界坐标系
    pv.pnt = ext.R * pv.pnt + ext.p;
    
    // 将方差矩阵转换到世界坐标系
    pv.var = ext.R * pv.var * ext.R.transpose();
  }
}

/**
 * @brief 更新点云中每个点的协方差矩阵并转换到世界坐标系
 * @param pptr 存储点云及其方差信息的指针
 * @param x_curr 当前IMU状态（包含位姿和协方差信息）
 * @param pwld 存储转换到世界坐标系的点云容器
 */
void pvec_update(PVecPtr pptr, IMUST &x_curr, PLV(3) &pwld)
{
  // 从IMU状态协方差矩阵中提取旋转部分的3x3协方差子矩阵
  Eigen::Matrix3d rot_var = x_curr.cov.block<3, 3>(0, 0);
  // 从IMU状态协方差矩阵中提取平移部分的3x3协方差子矩阵
  Eigen::Matrix3d tsl_var = x_curr.cov.block<3, 3>(3, 3);

  // 遍历每个点及其方差信息
  for(pointVar &pv: *pptr)
  {
    // 将点坐标转换为反对称矩阵（用于叉积运算）
    Eigen::Matrix3d phat = hat(pv.pnt);
    
    // 更新点的协方差矩阵，考虑三个不确定性来源：
    // 1. 原始点的不确定性：x_curr.R * pv.var * x_curr.R.transpose()
    // 2. 旋转不确定性的影响：phat * rot_var * phat.transpose()
    // 3. 平移不确定性的影响：tsl_var
    pv.var = x_curr.R * pv.var * x_curr.R.transpose() + phat * rot_var * phat.transpose() + tsl_var;
    
    // 将点从局部坐标系转换到世界坐标系：p_world = R * p_local + t
    // 并将转换后的点添加到世界坐标系点云容器中
    pwld.push_back(x_curr.R * pv.pnt + x_curr.p);
  }
}

// Read the alidarstate.txt
void read_lidarstate(string filename, vector<ScanPose*> &bl_tem)
{
  ifstream file(filename);
  if(!file.is_open())
  {
    printf("Error: %s not found\n", filename.c_str());
    exit(0);
  }

  string lineStr, str;
  vector<double> nums;
  while(getline(file, lineStr))
  {
    nums.clear();
    stringstream ss(lineStr);
    while(getline(ss, str, ' '))
      nums.push_back(stod(str));
    
    IMUST xx;
    xx.t = nums[0];
    xx.p << nums[1], nums[2], nums[3];
    xx.R = Eigen::Quaterniond(nums[7], nums[4], nums[5], nums[6]).matrix();

    if(nums.size() >= 20)
    {
      xx.v << nums[8], nums[9], nums[10];
      xx.bg << nums[11], nums[12], nums[13];
      xx.ba << nums[14], nums[15], nums[16];
      xx.g << nums[17], nums[18], nums[19];
    }

    ScanPose* blp = new ScanPose(xx, nullptr);
    bl_tem.push_back(blp);

    if(nums.size() >= 26)
      for(int i=0; i<6; i++) 
        blp->v6[i] = nums[i + 20];
  }
}

double get_memory()
{
  ifstream infile("/proc/self/status");
  double mem = -1;
  string lineStr, str;
  while(getline(infile, lineStr))
  {
    stringstream ss(lineStr);
    bool is_find = false;
    while(ss >> str)
    {
      if(str == "VmRSS:")
      {
        is_find = true; continue;
      }

      if(is_find) mem = stod(str);
      break;
    }
    if(is_find) break;
  }
  return mem / (1048576);
}

void icp_check(pcl::PointCloud<PointType> &pl_src, pcl::PointCloud<PointType> &pl_tar, ros::Publisher &pub_src, ros::Publisher &pub_tar, pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform, IMUST &xx)
{
  pcl::PointCloud<PointType> pl1, pl2;
  for(PointType ap: pl_src.points)
  {
    Eigen::Vector3d v(ap.x, ap.y, ap.z);
    v = loop_transform.second * v + loop_transform.first;
    v = xx.R * v + xx.p;
    ap.x = v[0]; ap.y = v[1]; ap.z = v[2];
    pl1.push_back(ap);
  }
  for(PointType ap: pl_tar.points)
  {
    Eigen::Vector3d v(ap.x, ap.y, ap.z);
    v = xx.R * v + xx.p;
    ap.x = v[0]; ap.y = v[1]; ap.z = v[2];
    pl2.push_back(ap);
  }
  pub_pl_func(pl1, pub_src); pub_pl_func(pl2, pub_tar);
}

