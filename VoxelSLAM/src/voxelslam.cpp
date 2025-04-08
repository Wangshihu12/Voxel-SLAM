#include "voxelslam.hpp"

using namespace std;

class ResultOutput
{
public:
  static ResultOutput &instance()
  {
    static ResultOutput inst;
    return inst;
  }

  void pub_odom_func(IMUST &xc)
  {
    Eigen::Quaterniond q_this(xc.R);
    Eigen::Vector3d t_this = xc.p;

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(t_this.x(), t_this.y(), t_this.z()));
    q.setW(q_this.w());
    q.setX(q_this.x());
    q.setY(q_this.y());
    q.setZ(q_this.z());
    transform.setRotation(q);
    ros::Time ct = ros::Time::now();
    br.sendTransform(tf::StampedTransform(transform, ct, "/camera_init", "/aft_mapped"));
  }

  void pub_localtraj(PLV(3) & pwld, double jour, IMUST &x_curr, int cur_session, pcl::PointCloud<PointType> &pcl_path)
  {
    pub_odom_func(x_curr);
    pcl::PointCloud<PointType> pcl_send;
    pcl_send.reserve(pwld.size());
    for (Eigen::Vector3d &pw : pwld)
    {
      Eigen::Vector3d pvec = pw;
      PointType ap;
      ap.x = pvec.x();
      ap.y = pvec.y();
      ap.z = pvec.z();
      pcl_send.push_back(ap);
    }
    pub_pl_func(pcl_send, pub_scan);

    Eigen::Vector3d pcurr = x_curr.p;

    PointType ap;
    ap.x = pcurr[0];
    ap.y = pcurr[1];
    ap.z = pcurr[2];
    ap.curvature = jour;
    ap.intensity = cur_session;
    pcl_path.push_back(ap);
    pub_pl_func(pcl_path, pub_curr_path);
  }

  void pub_localmap(int mgsize, int cur_session, vector<PVecPtr> &pvec_buf, vector<IMUST> &x_buf, pcl::PointCloud<PointType> &pcl_path, int win_base, int win_count)
  {
    pcl::PointCloud<PointType> pcl_send;
    for (int i = 0; i < mgsize; i++)
    {
      for (int j = 0; j < pvec_buf[i]->size(); j += 3)
      {
        pointVar &pv = pvec_buf[i]->at(j);
        Eigen::Vector3d pvec = x_buf[i].R * pv.pnt + x_buf[i].p;
        PointType ap;
        ap.x = pvec[0];
        ap.y = pvec[1];
        ap.z = pvec[2];
        ap.intensity = cur_session;
        pcl_send.push_back(ap);
      }
    }

    for (int i = 0; i < win_count; i++)
    {
      Eigen::Vector3d pcurr = x_buf[i].p;
      pcl_path[i + win_base].x = pcurr[0];
      pcl_path[i + win_base].y = pcurr[1];
      pcl_path[i + win_base].z = pcurr[2];
    }

    pub_pl_func(pcl_path, pub_curr_path);
    pub_pl_func(pcl_send, pub_cmap);
  }

  void pub_global_path(vector<vector<ScanPose *> *> &relc_bl_buf, ros::Publisher &pub_relc, vector<int> &ids)
  {
    pcl::PointCloud<pcl::PointXYZI> pl;
    pcl::PointXYZI pp;
    int idsize = ids.size();

    for (int i = 0; i < idsize; i++)
    {
      pp.intensity = ids[i];
      for (ScanPose *bl : *(relc_bl_buf[ids[i]]))
      {
        pp.x = bl->x.p[0];
        pp.y = bl->x.p[1];
        pp.z = bl->x.p[2];
        pl.push_back(pp);
      }
    }
    pub_pl_func(pl, pub_relc);
  }

  void pub_globalmap(vector<vector<Keyframe *> *> &relc_submaps, vector<int> &ids, ros::Publisher &pub)
  {
    pcl::PointCloud<pcl::PointXYZI> pl;
    pub_pl_func(pl, pub);
    pcl::PointXYZI pp;

    uint interval_size = 5e6;
    uint psize = 0;
    for (int id : ids)
    {
      vector<Keyframe *> &smps = *(relc_submaps[id]);
      for (int i = 0; i < smps.size(); i++)
        psize += smps[i]->plptr->size();
    }
    int jump = psize / (10 * interval_size) + 1;

    for (int id : ids)
    {
      pp.intensity = id;
      vector<Keyframe *> &smps = *(relc_submaps[id]);
      for (int i = 0; i < smps.size(); i++)
      {
        IMUST xx = smps[i]->x0;
        for (int j = 0; j < smps[i]->plptr->size(); j += jump)
        // for(int j=0; j<smps[i]->plptr->size(); j+=1)
        {
          PointType &ap = smps[i]->plptr->points[j];
          Eigen::Vector3d vv(ap.x, ap.y, ap.z);
          vv = xx.R * vv + xx.p;
          pp.x = vv[0];
          pp.y = vv[1];
          pp.z = vv[2];
          pl.push_back(pp);
        }

        if (pl.size() > interval_size)
        {
          pub_pl_func(pl, pub);
          sleep(0.05);
          pl.clear();
        }
      }
    }
    pub_pl_func(pl, pub);
  }
};

class FileReaderWriter
{
public:
  static FileReaderWriter &instance()
  {
    static FileReaderWriter inst;
    return inst;
  }

  void save_pcd(PVecPtr pptr, IMUST &xx, int count, const string &savename)
  {
    pcl::PointCloud<pcl::PointXYZI> pl_save;
    for (pointVar &pw : *pptr)
    {
      pcl::PointXYZI ap;
      ap.x = pw.pnt[0];
      ap.y = pw.pnt[1];
      ap.z = pw.pnt[2];
      pl_save.push_back(ap);
    }
    string pcdname = savename + "/" + to_string(count) + ".pcd";
    pcl::io::savePCDFileBinary(pcdname, pl_save);
  }

  void save_pose(vector<ScanPose *> &bbuf, string &fname, string posename, string &savepath)
  {
    if (bbuf.size() < 100)
      return;
    int topsize = bbuf.size();

    ofstream posfile(savepath + fname + posename);
    for (int i = 0; i < topsize; i++)
    {
      IMUST &xx = bbuf[i]->x;
      Eigen::Quaterniond qq(xx.R);
      posfile << fixed << setprecision(6) << xx.t << " ";
      posfile << setprecision(7) << xx.p[0] << " " << xx.p[1] << " " << xx.p[2] << " ";
      posfile << qq.x() << " " << qq.y() << " " << qq.z() << " " << qq.w();
      posfile << " " << xx.v[0] << " " << xx.v[1] << " " << xx.v[2];
      posfile << " " << xx.bg[0] << " " << xx.bg[1] << " " << xx.bg[2];
      posfile << " " << xx.ba[0] << " " << xx.ba[1] << " " << xx.ba[2];
      posfile << " " << xx.g[0] << " " << xx.g[1] << " " << xx.g[2];
      for (int j = 0; j < 6; j++)
        posfile << " " << bbuf[i]->v6[j];
      posfile << endl;
    }
    posfile.close();
  }

  // The loop clousure edges of multi sessions
  void pgo_edges_io(PGO_Edges &edges, vector<string> &fnames, int io, string &savepath, string &bagname)
  {
    static vector<string> seq_absent;
    Eigen::Matrix<double, 6, 1> v6_init;
    v6_init << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;
    if (io == 0) // read
    {
      ifstream infile(savepath + "edge.txt");
      string lineStr, str;
      vector<string> sts;
      while (getline(infile, lineStr))
      {
        sts.clear();
        stringstream ss(lineStr);
        while (ss >> str)
          sts.push_back(str);

        int mp[2] = {-1, -1};
        for (int i = 0; i < 2; i++)
          for (int j = 0; j < fnames.size(); j++)
            if (sts[i] == fnames[j])
            {
              mp[i] = j;
              break;
            }

        if (mp[0] != -1 && mp[1] != -1)
        {
          int id1 = stoi(sts[2]);
          int id2 = stoi(sts[3]);
          Eigen::Vector3d v3;
          v3 << stod(sts[4]), stod(sts[5]), stod(sts[6]);
          Eigen::Quaterniond qq(stod(sts[10]), stod(sts[7]), stod(sts[8]), stod(sts[9]));
          Eigen::Matrix3d rot(qq.matrix());
          if (mp[0] <= mp[1])
            edges.push(mp[0], mp[1], id1, id2, rot, v3, v6_init);
          else
          {
            v3 = -rot.transpose() * v3;
            rot = qq.matrix().transpose();
            edges.push(mp[1], mp[0], id2, id1, rot, v3, v6_init);
          }
        }
        else
        {
          if (sts[0] != bagname && sts[1] != bagname)
            seq_absent.push_back(lineStr);
        }
      }
    }
    else // write
    {
      ofstream outfile(savepath + "edge.txt");
      for (string &str : seq_absent)
        outfile << str << endl;

      for (PGO_Edge &edge : edges.edges)
      {
        for (int i = 0; i < edge.rots.size(); i++)
        {
          outfile << fnames[edge.m1] << " ";
          outfile << fnames[edge.m2] << " ";
          outfile << edge.ids1[i] << " ";
          outfile << edge.ids2[i] << " ";
          Eigen::Vector3d v(edge.tras[i]);
          outfile << setprecision(7) << v[0] << " " << v[1] << " " << v[2] << " ";
          Eigen::Quaterniond qq(edge.rots[i]);
          outfile << qq.x() << " " << qq.y() << " " << qq.z() << " " << qq.w() << endl;
        }
      }
      outfile.close();
    }
  }

  // loading the offline map
  void previous_map_names(ros::NodeHandle &n, vector<string> &fnames, vector<double> &juds)
  {
    string premap;
    n.param<string>("General/previous_map", premap, "");
    premap.erase(remove_if(premap.begin(), premap.end(), ::isspace), premap.end());
    stringstream ss(premap);
    string str;
    while (getline(ss, str, ','))
    {
      stringstream ss2(str);
      vector<string> strs;
      while (getline(ss2, str, ':'))
        strs.push_back(str);

      if (strs.size() != 2)
      {
        printf("previous map name wrong\n");
        return;
      }

      if (strs[0][0] != '#')
      {
        fnames.push_back(strs[0]);
        juds.push_back(stod(strs[1]));
      }
    }
  }

  void previous_map_read(vector<STDescManager *> &std_managers, vector<vector<ScanPose *> *> &multimap_scanPoses, vector<vector<Keyframe *> *> &multimap_keyframes, ConfigSetting &config_setting, PGO_Edges &edges, ros::NodeHandle &n, vector<string> &fnames, vector<double> &juds, string &savepath, int win_size)
  {
    int acsize = 10;
    int mgsize = 5;
    n.param<int>("Loop/acsize", acsize, 10);
    n.param<int>("Loop/mgsize", mgsize, 5);

    for (int fn = 0; fn < fnames.size() && n.ok(); fn++)
    {
      string fname = savepath + fnames[fn];
      vector<ScanPose *> *bl_tem = new vector<ScanPose *>();
      vector<Keyframe *> *keyframes_tem = new vector<Keyframe *>();
      STDescManager *std_manager = new STDescManager(config_setting);

      std_managers.push_back(std_manager);
      multimap_scanPoses.push_back(bl_tem);
      multimap_keyframes.push_back(keyframes_tem);
      read_lidarstate(fname + "/alidarState.txt", *bl_tem);

      cout << "Reading " << fname << ": " << bl_tem->size() << " scans." << "\n";
      deque<pcl::PointCloud<pcl::PointXYZI>::Ptr> plbuf;
      deque<IMUST> xxbuf;
      pcl::PointCloud<PointType> pl_lc;
      pcl::PointCloud<pcl::PointXYZI>::Ptr pl_btc(new pcl::PointCloud<pcl::PointXYZI>());

      for (int i = 0; i < bl_tem->size() && n.ok(); i++)
      {
        IMUST &xc = bl_tem->at(i)->x;
        string pcdname = fname + "/" + to_string(i) + ".pcd";
        pcl::PointCloud<pcl::PointXYZI>::Ptr pl_tem(new pcl::PointCloud<pcl::PointXYZI>());
        pcl::io::loadPCDFile(pcdname, *pl_tem);

        xxbuf.push_back(xc);
        plbuf.push_back(pl_tem);

        if (xxbuf.size() < win_size)
          continue;

        pl_lc.clear();
        Keyframe *smp = new Keyframe(xc);
        smp->id = i;
        PointType pt;
        for (int j = 0; j < win_size; j++)
        {
          Eigen::Vector3d delta_p = xc.R.transpose() * (xxbuf[j].p - xc.p);
          Eigen::Matrix3d delta_R = xc.R.transpose() * xxbuf[j].R;

          for (pcl::PointXYZI pp : plbuf[j]->points)
          {
            Eigen::Vector3d v3(pp.x, pp.y, pp.z);
            v3 = delta_R * v3 + delta_p;
            pt.x = v3[0];
            pt.y = v3[1];
            pt.z = v3[2];
            pl_lc.push_back(pt);
          }
        }

        down_sampling_voxel(pl_lc, voxel_size / 10);
        smp->plptr->reserve(pl_lc.size());
        for (PointType &pp : pl_lc.points)
          smp->plptr->push_back(pp);
        keyframes_tem->push_back(smp);

        for (int j = 0; j < win_size; j++)
        {
          plbuf.pop_front();
          xxbuf.pop_front();
        }
      }

      cout << "Generating BTC descriptors..." << "\n";

      int subsize = keyframes_tem->size();
      for (int i = 0; i + acsize < subsize && n.ok(); i += mgsize)
      {
        int up = i + acsize;
        pl_btc->clear();
        IMUST &xc = keyframes_tem->at(up - 1)->x0;
        for (int j = i; j < up; j++)
        {
          IMUST &xj = keyframes_tem->at(j)->x0;
          Eigen::Vector3d delta_p = xc.R.transpose() * (xj.p - xc.p);
          Eigen::Matrix3d delta_R = xc.R.transpose() * xj.R;
          pcl::PointXYZI pp;
          for (PointType ap : keyframes_tem->at(j)->plptr->points)
          {
            Eigen::Vector3d v3(ap.x, ap.y, ap.z);
            v3 = delta_R * v3 + delta_p;
            pp.x = v3[0];
            pp.y = v3[1];
            pp.z = v3[2];
            pl_btc->push_back(pp);
          }
        }

        vector<STD> stds_vec;
        std_manager->GenerateSTDescs(pl_btc, stds_vec, keyframes_tem->at(up - 1)->id);
        std_manager->AddSTDescs(stds_vec);
      }
      std_manager->config_setting_.skip_near_num_ = -(std_manager->plane_cloud_vec_.size() + 10);

      cout << "Read " << fname << " done." << "\n\n";
    }

    vector<int> ids_all;
    for (int fn = 0; fn < fnames.size() && n.ok(); fn++)
      ids_all.push_back(fn);

    // gtsam::Values initial;
    // gtsam::NonlinearFactorGraph graph;
    // vector<int> ids_cnct, stepsizes;
    // Eigen::Matrix<double, 6, 1> v6_init;
    // v6_init << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    // gtsam::noiseModel::Diagonal::shared_ptr odom_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector(v6_init));
    // build_graph(initial, graph, ids_all.back(), edges, odom_noise, ids_cnct, stepsizes, 1);

    // gtsam::ISAM2Params parameters;
    // parameters.relinearizeThreshold = 0.01;
    // parameters.relinearizeSkip = 1;
    // gtsam::ISAM2 isam(parameters);
    // isam.update(graph, initial);

    // for(int i=0; i<5; i++) isam.update();
    // gtsam::Values results = isam.calculateEstimate();
    // int resultsize = results.size();
    // int idsize = ids_cnct.size();
    // for(int ii=0; ii<idsize; ii++)
    // {
    //   int tip = ids_cnct[ii];
    //   for(int j=stepsizes[ii]; j<stepsizes[ii+1]; j++)
    //   {
    //     int ord = j - stepsizes[ii];
    //     multimap_scanPoses[tip]->at(ord)->set_state(results.at(j).cast<gtsam::Pose3>());
    //   }
    // }
    // for(int ii=0; ii<idsize; ii++)
    // {
    //   int tip = ids_cnct[ii];
    //   for(Keyframe *kf: *multimap_keyframes[tip])
    //     kf->x0 = multimap_scanPoses[tip]->at(kf->id)->x;
    // }

    ResultOutput::instance().pub_global_path(multimap_scanPoses, pub_prev_path, ids_all);
    ResultOutput::instance().pub_globalmap(multimap_keyframes, ids_all, pub_pmap);

    printf("All the maps are loaded\n");
  }
};

class Initialization
{
public:
  static Initialization &instance()
  {
    static Initialization inst;
    return inst;
  }

  void align_gravity(vector<IMUST> &xs)
  {
    Eigen::Vector3d g0 = xs[0].g;
    Eigen::Vector3d n0 = g0 / g0.norm();
    Eigen::Vector3d n1(0, 0, 1);
    if (n0[2] < 0)
      n1[2] = -1;

    Eigen::Vector3d rotvec = n0.cross(n1);
    double rnorm = rotvec.norm();
    rotvec = rotvec / rnorm;

    Eigen::AngleAxisd angaxis(asin(rnorm), rotvec);
    Eigen::Matrix3d rot = angaxis.matrix();
    g0 = rot * g0;

    Eigen::Vector3d p0 = xs[0].p;
    for (int i = 0; i < xs.size(); i++)
    {
      xs[i].p = rot * (xs[i].p - p0) + p0;
      xs[i].R = rot * xs[i].R;
      xs[i].v = rot * xs[i].v;
      xs[i].g = g0;
    }
  }

  void motion_blur(pcl::PointCloud<PointType> &pl, PVec &pvec, IMUST xc, IMUST xl, deque<sensor_msgs::Imu::Ptr> &imus, double pcl_beg_time, IMUST &extrin_para)
  {
    xc.bg = xl.bg;
    xc.ba = xl.ba;
    Eigen::Vector3d acc_imu, angvel_avr, acc_avr, vel_imu(xc.v), pos_imu(xc.p);
    Eigen::Matrix3d R_imu(xc.R);
    vector<IMUST> imu_poses;

    for (auto it_imu = imus.end() - 1; it_imu != imus.begin(); it_imu--)
    {
      sensor_msgs::Imu &head = **(it_imu - 1);
      sensor_msgs::Imu &tail = **(it_imu);

      angvel_avr << 0.5 * (head.angular_velocity.x + tail.angular_velocity.x),
          0.5 * (head.angular_velocity.y + tail.angular_velocity.y),
          0.5 * (head.angular_velocity.z + tail.angular_velocity.z);
      acc_avr << 0.5 * (head.linear_acceleration.x + tail.linear_acceleration.x),
          0.5 * (head.linear_acceleration.y + tail.linear_acceleration.y),
          0.5 * (head.linear_acceleration.z + tail.linear_acceleration.z);

      angvel_avr -= xc.bg;
      acc_avr = acc_avr * imupre_scale_gravity - xc.ba;

      double dt = head.header.stamp.toSec() - tail.header.stamp.toSec();
      Eigen::Matrix3d acc_avr_skew = hat(acc_avr);
      Eigen::Matrix3d Exp_f = Exp(angvel_avr, dt);

      acc_imu = R_imu * acc_avr + xc.g;
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;
      vel_imu = vel_imu + acc_imu * dt;
      R_imu = R_imu * Exp_f;

      double offt = head.header.stamp.toSec() - pcl_beg_time;
      imu_poses.emplace_back(offt, R_imu, pos_imu, vel_imu, angvel_avr, acc_imu);
    }

    pointVar pv;
    pv.var.setIdentity();
    if (point_notime)
    {
      for (PointType &ap : pl.points)
      {
        pv.pnt << ap.x, ap.y, ap.z;
        pv.pnt = extrin_para.R * pv.pnt + extrin_para.p;
        pvec.push_back(pv);
      }
      return;
    }
    auto it_pcl = pl.end() - 1;
    // for(auto it_kp=imu_poses.end(); it_kp!=imu_poses.begin(); it_kp--)
    for (auto it_kp = imu_poses.begin(); it_kp != imu_poses.end(); it_kp++)
    {
      // IMUST &head = *(it_kp - 1);
      IMUST &head = *it_kp;
      R_imu = head.R;
      acc_imu = head.ba;
      vel_imu = head.v;
      pos_imu = head.p;
      angvel_avr = head.bg;

      for (; it_pcl->curvature > head.t; it_pcl--)
      {
        double dt = it_pcl->curvature - head.t;
        Eigen::Matrix3d R_i = R_imu * Exp(angvel_avr, dt);
        Eigen::Vector3d T_ei = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - xc.p;

        Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        Eigen::Vector3d P_compensate = xc.R.transpose() * (R_i * (extrin_para.R * P_i + extrin_para.p) + T_ei);

        pv.pnt = P_compensate;
        pvec.push_back(pv);
        if (it_pcl == pl.begin())
          break;
      }
    }
  }

  int motion_init(vector<pcl::PointCloud<PointType>::Ptr> &pl_origs, vector<deque<sensor_msgs::Imu::Ptr>> &vec_imus, vector<double> &beg_times, Eigen::MatrixXd *hess, LidarFactor &voxhess, vector<IMUST> &x_buf, unordered_map<VOXEL_LOC, OctoTree *> &surf_map, unordered_map<VOXEL_LOC, OctoTree *> &surf_map_slide, vector<PVecPtr> &pvec_buf, int win_size, vector<vector<SlideWindow *>> &sws, IMUST &x_curr, deque<IMU_PRE *> &imu_pre_buf, IMUST &extrin_para)
  {
    PLV(3)
    pwld;
    double last_g_norm = x_buf[0].g.norm();
    int converge_flag = 0;

    double min_eigen_value_orig = min_eigen_value;
    vector<double> eigen_value_array_orig = plane_eigen_value_thre;

    min_eigen_value = 0.02;
    for (double &iter : plane_eigen_value_thre)
      iter = 1.0 / 4;

    double t0 = ros::Time::now().toSec();
    double converge_thre = 0.05;
    int converge_times = 0;
    bool is_degrade = true;
    Eigen::Vector3d eigvalue;
    eigvalue.setZero();
    for (int iterCnt = 0; iterCnt < 10; iterCnt++)
    {
      if (converge_flag == 1)
      {
        min_eigen_value = min_eigen_value_orig;
        plane_eigen_value_thre = eigen_value_array_orig;
      }

      vector<OctoTree *> octos;
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      {
        iter->second->tras_ptr(octos);
        iter->second->clear_slwd(sws[0]);
        delete iter->second;
      }
      for (int i = 0; i < octos.size(); i++)
        delete octos[i];
      surf_map.clear();
      octos.clear();
      surf_map_slide.clear();

      for (int i = 0; i < win_size; i++)
      {
        pwld.clear();
        pvec_buf[i]->clear();
        int l = i == 0 ? i : i - 1;
        motion_blur(*pl_origs[i], *pvec_buf[i], x_buf[i], x_buf[l], vec_imus[i], beg_times[i], extrin_para);

        if (converge_flag == 1)
        {
          for (pointVar &pv : *pvec_buf[i])
            calcBodyVar(pv.pnt, dept_err, beam_err, pv.var);
          pvec_update(pvec_buf[i], x_buf[i], pwld);
        }
        else
        {
          for (pointVar &pv : *pvec_buf[i])
            pwld.push_back(x_buf[i].R * pv.pnt + x_buf[i].p);
        }

        cut_voxel(surf_map, pvec_buf[i], i, surf_map_slide, win_size, pwld, sws[0]);
      }

      // LidarFactor voxhess(win_size);
      voxhess.clear();
      voxhess.win_size = win_size;
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      {
        iter->second->recut(win_size, x_buf, sws[0]);
        iter->second->tras_opt(voxhess);
      }

      if (voxhess.plvec_voxels.size() < 10)
        break;
      LI_BA_OptimizerGravity opt_lsv;
      vector<double> resis;
      opt_lsv.damping_iter(x_buf, voxhess, imu_pre_buf, resis, hess, 3);
      Eigen::Matrix3d nnt;
      nnt.setZero();

      printf("%d: %lf %lf %lf: %lf %lf\n", iterCnt, x_buf[0].g[0], x_buf[0].g[1], x_buf[0].g[2], x_buf[0].g.norm(), fabs(resis[0] - resis[1]) / resis[0]);

      for (int i = 0; i < win_size - 1; i++)
        delete imu_pre_buf[i];
      imu_pre_buf.clear();

      for (int i = 1; i < win_size; i++)
      {
        imu_pre_buf.push_back(new IMU_PRE(x_buf[i - 1].bg, x_buf[i - 1].ba));
        imu_pre_buf.back()->push_imu(vec_imus[i]);
      }

      if (fabs(resis[0] - resis[1]) / resis[0] < converge_thre && iterCnt >= 2)
      {
        for (Eigen::Matrix3d &iter : voxhess.eig_vectors)
        {
          Eigen::Vector3d v3 = iter.col(0);
          nnt += v3 * v3.transpose();
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(nnt);
        eigvalue = saes.eigenvalues();
        is_degrade = eigvalue[0] < 15 ? true : false;

        converge_thre = 0.01;
        if (converge_flag == 0)
        {
          align_gravity(x_buf);
          converge_flag = 1;
          continue;
        }
        else
          break;
      }
    }

    x_curr = x_buf[win_size - 1];
    double gnm = x_curr.g.norm();
    if (is_degrade || gnm < 9.6 || gnm > 10.0)
    {
      converge_flag = 0;
    }
    if (converge_flag == 0)
    {
      vector<OctoTree *> octos;
      for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      {
        iter->second->tras_ptr(octos);
        iter->second->clear_slwd(sws[0]);
        delete iter->second;
      }
      for (int i = 0; i < octos.size(); i++)
        delete octos[i];
      surf_map.clear();
      octos.clear();
      surf_map_slide.clear();
    }

    printf("mn: %lf %lf %lf\n", eigvalue[0], eigvalue[1], eigvalue[2]);
    Eigen::Vector3d angv(vec_imus[0][0]->angular_velocity.x, vec_imus[0][0]->angular_velocity.y, vec_imus[0][0]->angular_velocity.z);
    Eigen::Vector3d acc(vec_imus[0][0]->linear_acceleration.x, vec_imus[0][0]->linear_acceleration.y, vec_imus[0][0]->linear_acceleration.z);
    acc *= 9.8;

    pl_origs.clear();
    vec_imus.clear();
    beg_times.clear();
    double t1 = ros::Time::now().toSec();
    printf("init time: %lf\n", t1 - t0);

    // align_gravity(x_buf);
    pcl::PointCloud<PointType> pcl_send;
    PointType pt;
    for (int i = 0; i < win_size; i++)
      for (pointVar &pv : *pvec_buf[i])
      {
        Eigen::Vector3d vv = x_buf[i].R * pv.pnt + x_buf[i].p;
        pt.x = vv[0];
        pt.y = vv[1];
        pt.z = vv[2];
        pcl_send.push_back(pt);
      }
    pub_pl_func(pcl_send, pub_init);

    return converge_flag;
  }
};

class VOXEL_SLAM
{
public:
  pcl::PointCloud<PointType> pcl_path;
  IMUST x_curr, extrin_para;
  IMUEKF odom_ekf;
  unordered_map<VOXEL_LOC, OctoTree *> surf_map, surf_map_slide;
  double down_size;

  int win_size;
  vector<IMUST> x_buf;
  vector<PVecPtr> pvec_buf;
  deque<IMU_PRE *> imu_pre_buf;
  int win_count = 0, win_base = 0;
  vector<vector<SlideWindow *>> sws;

  vector<ScanPose *> *scanPoses;
  mutex mtx_loop;
  deque<ScanPose *> buf_lba2loop, buf_lba2loop_tem;
  vector<Keyframe *> *keyframes;
  int loop_detect = 0;
  unordered_map<VOXEL_LOC, OctoTree *> map_loop;
  IMUST dx;
  pcl::PointCloud<PointType>::Ptr pl_kdmap;
  pcl::KdTreeFLANN<PointType> kd_keyframes;
  int history_kfsize = 0;
  vector<OctoTree *> octos_release;
  int reset_flag = 0;
  int g_update = 0;
  int thread_num = 5;
  int degrade_bound = 10;

  vector<vector<ScanPose *> *> multimap_scanPoses;
  vector<vector<Keyframe *> *> multimap_keyframes;
  volatile int gba_flag = 0;
  int gba_size = 0;
  vector<int> cnct_map;
  mutex mtx_keyframe;
  PGO_Edges gba_edges1, gba_edges2;
  bool is_finish = false;

  vector<string> sessionNames;
  string bagname, savepath;
  int is_save_map;

  VOXEL_SLAM(ros::NodeHandle &n)
  {
    double cov_gyr, cov_acc, rand_walk_gyr, rand_walk_acc;
    vector<double> vecR(9), vecT(3);
    scanPoses = new vector<ScanPose *>();
    keyframes = new vector<Keyframe *>();

    string lid_topic, imu_topic;
    n.param<string>("General/lid_topic", lid_topic, "/livox/lidar");
    n.param<string>("General/imu_topic", imu_topic, "/livox/imu");
    n.param<string>("General/bagname", bagname, "site3_handheld_4");
    n.param<string>("General/save_path", savepath, "");
    n.param<int>("General/lidar_type", feat.lidar_type, 0);
    n.param<double>("General/blind", feat.blind, 0.1);
    n.param<int>("General/point_filter_num", feat.point_filter_num, 3);
    n.param<vector<double>>("General/extrinsic_tran", vecT, vector<double>());
    n.param<vector<double>>("General/extrinsic_rota", vecR, vector<double>());
    n.param<int>("General/is_save_map", is_save_map, 0);

    sub_imu = n.subscribe(imu_topic, 80000, imu_handler);
    if (feat.lidar_type == LIVOX)
      sub_pcl = n.subscribe<livox_ros_driver::CustomMsg>(lid_topic, 1000, pcl_handler);
    else
      sub_pcl = n.subscribe<sensor_msgs::PointCloud2>(lid_topic, 1000, pcl_handler);
    odom_ekf.imu_topic = imu_topic;

    n.param<double>("Odometry/cov_gyr", cov_gyr, 0.1);
    n.param<double>("Odometry/cov_acc", cov_acc, 0.1);
    n.param<double>("Odometry/rdw_gyr", rand_walk_gyr, 1e-4);
    n.param<double>("Odometry/rdw_acc", rand_walk_acc, 1e-4);
    n.param<double>("Odometry/down_size", down_size, 0.1);
    n.param<double>("Odometry/dept_err", dept_err, 0.02);
    n.param<double>("Odometry/beam_err", beam_err, 0.05);
    n.param<double>("Odometry/voxel_size", voxel_size, 1);
    n.param<double>("Odometry/min_eigen_value", min_eigen_value, 0.0025);
    n.param<int>("Odometry/degrade_bound", degrade_bound, 10);
    n.param<int>("Odometry/point_notime", point_notime, 0);
    odom_ekf.point_notime = point_notime;

    feat.blind = feat.blind * feat.blind;
    odom_ekf.cov_gyr << cov_gyr, cov_gyr, cov_gyr;
    odom_ekf.cov_acc << cov_acc, cov_acc, cov_acc;
    odom_ekf.cov_bias_gyr << rand_walk_gyr, rand_walk_gyr, rand_walk_gyr;
    odom_ekf.cov_bias_acc << rand_walk_acc, rand_walk_acc, rand_walk_acc;
    odom_ekf.Lid_offset_to_IMU << vecT[0], vecT[1], vecT[2];
    odom_ekf.Lid_rot_to_IMU << vecR[0], vecR[1], vecR[2],
        vecR[3], vecR[4], vecR[5],
        vecR[6], vecR[7], vecR[8];
    extrin_para.R = odom_ekf.Lid_rot_to_IMU;
    extrin_para.p = odom_ekf.Lid_offset_to_IMU;
    min_point << 5, 5, 5, 5;

    n.param<int>("LocalBA/win_size", win_size, 10);
    n.param<int>("LocalBA/max_layer", max_layer, 2);
    n.param<double>("LocalBA/cov_gyr", cov_gyr, 0.1);
    n.param<double>("LocalBA/cov_acc", cov_acc, 0.1);
    n.param<double>("LocalBA/rdw_gyr", rand_walk_gyr, 1e-4);
    n.param<double>("LocalBA/rdw_acc", rand_walk_acc, 1e-4);
    n.param<int>("LocalBA/min_ba_point", min_ba_point, 20);
    n.param<vector<double>>("LocalBA/plane_eigen_value_thre", plane_eigen_value_thre, vector<double>({1, 1, 1, 1}));
    n.param<double>("LocalBA/imu_coef", imu_coef, 1e-4);
    n.param<int>("LocalBA/thread_num", thread_num, 5);

    for (double &iter : plane_eigen_value_thre)
      iter = 1.0 / iter;
    // for(double &iter: plane_eigen_value_thre) iter = 1.0 / iter;

    noiseMeas.setZero();
    noiseWalk.setZero();
    noiseMeas.diagonal() << cov_gyr, cov_gyr, cov_gyr,
        cov_acc, cov_acc, cov_acc;
    noiseWalk.diagonal() << rand_walk_gyr, rand_walk_gyr, rand_walk_gyr,
        rand_walk_acc, rand_walk_acc, rand_walk_acc;

    int ss = 0;
    if (access((savepath + bagname + "/").c_str(), X_OK) == -1)
    {
      string cmd = "mkdir " + savepath + bagname + "/";
      ss = system(cmd.c_str());
    }
    else
      ss = -1;

    if (ss != 0 && is_save_map == 1)
    {
      printf("The pointcloud will be saved in this run.\n");
      printf("So please clear or rename the existed folder.\n");
      exit(0);
    }

    sws.resize(thread_num);
    cout << "bagname: " << bagname << endl;
  }

  // The point-to-plane alignment for odometry
  bool lio_state_estimation(PVecPtr pptr)
  {
    // 保存当前状态作为初始值
    IMUST x_prop = x_curr;

    // 设置最大迭代次数
    const int num_max_iter = 4;
    // 定义EKF停止和收敛标志
    bool EKF_stop_flg = 0, flg_EKF_converged = 0;
    // 定义状态矩阵
    Eigen::Matrix<double, DIM, DIM> G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();
    // 重匹配次数和匹配点数统计
    int rematch_num = 0;
    int match_num = 0;

    // 获取点云大小并初始化八叉树容器
    int psize = pptr->size();
    vector<OctoTree *> octos;
    octos.resize(psize, nullptr);

    // 初始化协方差矩阵
    Eigen::Matrix3d nnt;
    Eigen::Matrix<double, DIM, DIM> cov_inv = x_curr.cov.inverse();
    
    // 开始迭代优化
    for (int iterCount = 0; iterCount < num_max_iter; iterCount++)
    {
      // 初始化Hessian矩阵和残差向量
      Eigen::Matrix<double, 6, 6> HTH;
      HTH.setZero();
      Eigen::Matrix<double, 6, 1> HTz;
      HTz.setZero();
      
      // 获取旋转和平移的协方差
      Eigen::Matrix3d rot_var = x_curr.cov.block<3, 3>(0, 0);
      Eigen::Matrix3d tsl_var = x_curr.cov.block<3, 3>(3, 3);
      match_num = 0;
      nnt.setZero();

      // 遍历所有点进行点面匹配
      for (int i = 0; i < psize; i++)
      {
        pointVar &pv = pptr->at(i);
        // 计算点的斜对称矩阵
        Eigen::Matrix3d phat = hat(pv.pnt);
        // 计算点在全局坐标系下的协方差
        Eigen::Matrix3d var_world = x_curr.R * pv.var * x_curr.R.transpose() + phat * rot_var * phat.transpose() + tsl_var;
        // 将点转换到全局坐标系
        Eigen::Vector3d wld = x_curr.R * pv.pnt + x_curr.p;

        double sigma_d = 0;
        Plane *pla = nullptr;
        int flag = 0;
        
        // 尝试在八叉树中匹配平面
        if (octos[i] != nullptr && octos[i]->inside(wld))
        {
          double max_prob = 0;
          flag = octos[i]->match(wld, pla, max_prob, var_world, sigma_d, octos[i]);
        }
        else
        {
          flag = match(surf_map, wld, pla, var_world, sigma_d, octos[i]);
        }

        // 如果找到匹配的平面，计算残差和雅可比矩阵
        if (flag)
        {
          Plane &pp = *pla;
          // 计算信息矩阵
          double R_inv = 1.0 / (0.0005 + sigma_d);
          // 计算点到平面的距离残差
          double resi = pp.normal.dot(wld - pp.center);

          // 计算雅可比矩阵
          Eigen::Matrix<double, 6, 1> jac;
          jac.head(3) = phat * x_curr.R.transpose() * pp.normal;
          jac.tail(3) = pp.normal;
          
          // 更新Hessian矩阵和残差向量
          HTH += R_inv * jac * jac.transpose();
          HTz -= R_inv * jac * resi;
          nnt += pp.normal * pp.normal.transpose();
          match_num++;
        }
      }

      // 更新状态估计
      H_T_H.block<6, 6>(0, 0) = HTH;
      Eigen::Matrix<double, DIM, DIM> K_1 = (H_T_H + cov_inv).inverse();
      G.block<DIM, 6>(0, 0) = K_1.block<DIM, 6>(0, 0) * HTH;
      Eigen::Matrix<double, DIM, 1> vec = x_prop - x_curr;
      Eigen::Matrix<double, DIM, 1> solution = K_1.block<DIM, 6>(0, 0) * HTz + vec - G.block<DIM, 6>(0, 0) * vec.block<6, 1>(0, 0);

      // 更新当前状态
      x_curr += solution;
      Eigen::Vector3d rot_add = solution.block<3, 1>(0, 0);
      Eigen::Vector3d tra_add = solution.block<3, 1>(3, 0);

      // 检查收敛条件
      EKF_stop_flg = false;
      flg_EKF_converged = false;

      // 如果旋转和平移变化都很小，认为收敛
      if ((rot_add.norm() * 57.3 < 0.01) && (tra_add.norm() * 100 < 0.015))
        flg_EKF_converged = true;

      // 在收敛或最后一次迭代前进行重匹配
      if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == num_max_iter - 2)))
      {
        rematch_num++;
      }

      // 如果重匹配次数达到2次或达到最大迭代次数，更新协方差并停止
      if (rematch_num >= 2 || (iterCount == num_max_iter - 1))
      {
        x_curr.cov = (I_STATE - G) * x_curr.cov;
        EKF_stop_flg = true;
      }

      if (EKF_stop_flg)
        break;
    }

    // 检查优化结果的质量
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(nnt);
    Eigen::Vector3d evalue = saes.eigenvalues();

    // 如果最小特征值太小，认为优化失败
    if (evalue[0] < 14)
      return false;
    else
      return true;
  }

  // The point-to-plane alignment for initialization
  pcl::PointCloud<PointType>::Ptr pl_tree;
  void lio_state_estimation_kdtree(PVecPtr pptr)
  {
    static pcl::KdTreeFLANN<PointType> kd_map;
    if (pl_tree->size() < 100)
    {
      for (pointVar pv : *pptr)
      {
        PointType pp;
        pv.pnt = x_curr.R * pv.pnt + x_curr.p;
        pp.x = pv.pnt[0];
        pp.y = pv.pnt[1];
        pp.z = pv.pnt[2];
        pl_tree->push_back(pp);
      }
      kd_map.setInputCloud(pl_tree);
      return;
    }

    const int num_max_iter = 4;
    IMUST x_prop = x_curr;
    int psize = pptr->size();
    bool EKF_stop_flg = 0, flg_EKF_converged = 0;
    Eigen::Matrix<double, DIM, DIM> G, H_T_H, I_STATE;
    G.setZero();
    H_T_H.setZero();
    I_STATE.setIdentity();

    double max_dis = 2 * 2;
    vector<float> sqdis(NMATCH);
    vector<int> nearInd(NMATCH);
    PLV(3)
    vecs(NMATCH);
    int rematch_num = 0;
    Eigen::Matrix<double, DIM, DIM> cov_inv = x_curr.cov.inverse();

    Eigen::Matrix<double, NMATCH, 1> b;
    b.setOnes();
    b *= -1.0f;

    vector<double> ds(psize, -1);
    PLV(3)
    directs(psize);
    bool refind = true;

    for (int iterCount = 0; iterCount < num_max_iter; iterCount++)
    {
      Eigen::Matrix<double, 6, 6> HTH;
      HTH.setZero();
      Eigen::Matrix<double, 6, 1> HTz;
      HTz.setZero();
      int valid = 0;
      for (int i = 0; i < psize; i++)
      {
        pointVar &pv = pptr->at(i);
        Eigen::Matrix3d phat = hat(pv.pnt);
        Eigen::Vector3d wld = x_curr.R * pv.pnt + x_curr.p;

        if (refind)
        {
          PointType apx;
          apx.x = wld[0];
          apx.y = wld[1];
          apx.z = wld[2];
          kd_map.nearestKSearch(apx, NMATCH, nearInd, sqdis);

          Eigen::Matrix<double, NMATCH, 3> A;
          for (int i = 0; i < NMATCH; i++)
          {
            PointType &pp = pl_tree->points[nearInd[i]];
            A.row(i) << pp.x, pp.y, pp.z;
          }
          Eigen::Vector3d direct = A.colPivHouseholderQr().solve(b);
          bool check_flag = false;
          for (int i = 0; i < NMATCH; i++)
          {
            if (fabs(direct.dot(A.row(i)) + 1.0) > 0.1)
              check_flag = true;
          }

          if (check_flag)
          {
            ds[i] = -1;
            continue;
          }

          double d = 1.0 / direct.norm();
          // direct *= d;
          ds[i] = d;
          directs[i] = direct * d;
        }

        if (ds[i] >= 0)
        {
          double pd2 = directs[i].dot(wld) + ds[i];
          Eigen::Matrix<double, 6, 1> jac_s;
          jac_s.head(3) = phat * x_curr.R.transpose() * directs[i];
          jac_s.tail(3) = directs[i];

          HTH += jac_s * jac_s.transpose();
          HTz += jac_s * (-pd2);
          valid++;
        }
      }

      H_T_H.block<6, 6>(0, 0) = HTH;
      Eigen::Matrix<double, DIM, DIM> K_1 = (H_T_H + cov_inv / 1000).inverse();
      G.block<DIM, 6>(0, 0) = K_1.block<DIM, 6>(0, 0) * HTH;
      Eigen::Matrix<double, DIM, 1> vec = x_prop - x_curr;
      Eigen::Matrix<double, DIM, 1> solution = K_1.block<DIM, 6>(0, 0) * HTz + vec - G.block<DIM, 6>(0, 0) * vec.block<6, 1>(0, 0);

      x_curr += solution;
      Eigen::Vector3d rot_add = solution.block<3, 1>(0, 0);
      Eigen::Vector3d tra_add = solution.block<3, 1>(3, 0);

      refind = false;
      if ((rot_add.norm() * 57.3 < 0.01) && (tra_add.norm() * 100 < 0.015))
      {
        refind = true;
        flg_EKF_converged = true;
        rematch_num++;
      }

      if (iterCount == num_max_iter - 2 && !flg_EKF_converged)
      {
        refind = true;
      }

      if (rematch_num >= 2 || (iterCount == num_max_iter - 1))
      {
        x_curr.cov = (I_STATE - G) * x_curr.cov;
        EKF_stop_flg = true;
      }

      if (EKF_stop_flg)
        break;
    }

    double tt1 = ros::Time::now().toSec();
    for (pointVar pv : *pptr)
    {
      pv.pnt = x_curr.R * pv.pnt + x_curr.p;
      PointType ap;
      ap.x = pv.pnt[0];
      ap.y = pv.pnt[1];
      ap.z = pv.pnt[2];
      pl_tree->push_back(ap);
    }
    down_sampling_voxel(*pl_tree, 0.5);
    kd_map.setInputCloud(pl_tree);
    double tt2 = ros::Time::now().toSec();
  }

  // After detecting loop closure, refine current map and states
  void loop_update()
  {
    // 打印当前滑动窗口大小
    printf("loop update: %zu\n", sws[0].size());
    double t1 = ros::Time::now().toSec();

    // 清理当前表面特征地图
    for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
    {
      // 将八叉树节点转移到释放队列
      iter->second->tras_ptr(octos_release);
      // 清除滑动窗口中的特征
      iter->second->clear_slwd(sws[0]);
      // 删除八叉树节点
      delete iter->second;
      iter->second = nullptr;
    }
    // 清空当前地图和滑动窗口地图
    surf_map.clear();
    surf_map_slide.clear();
    // 将回环地图复制到当前地图
    surf_map = map_loop;
    map_loop.clear();

    // 打印各种数据结构的大小
    printf("scanPoses: %zu %zu %zu %d %d %zu\n", scanPoses->size(), buf_lba2loop.size(), x_buf.size(), win_base, win_count, sws[0].size());
    int blsize = scanPoses->size(); // 获取历史扫描位姿数量
    // 保存第一个点用于后续更新
    PointType ap = pcl_path[0]; // 保存第一个点作为模板
    pcl_path.clear(); // 清空路径点

    // 更新回环前的路径点
    for (int i = 0; i < blsize; i++)
    {
      ap.x = scanPoses->at(i)->x.p[0];
      ap.y = scanPoses->at(i)->x.p[1];
      ap.z = scanPoses->at(i)->x.p[2];
      pcl_path.push_back(ap);
    }

    // 更新回环检测到的位姿
    for (ScanPose *bl : buf_lba2loop)
    {
      // 应用回环修正
      bl->update(dx); // 应用回环修正变换dx到位姿
      ap.x = bl->x.p[0];
      ap.y = bl->x.p[1];
      ap.z = bl->x.p[2];
      pcl_path.push_back(ap);
    }

    // 更新当前窗口内的位姿
    for (int i = 0; i < win_count; i++)
    {
      IMUST &x = x_buf[i];
      // 应用回环修正到速度、位置和旋转
      x.v = dx.R * x.v;
      x.p = dx.R * x.p + dx.p;
      x.R = dx.R * x.R;
      // 如果需要更新重力向量
      if (g_update == 1)
        x.g = dx.R * x.g;
      // 更新路径点
      ap.x = x.p[0];
      ap.y = x.p[1];
      ap.z = x.p[2];
      pcl_path.push_back(ap);
    }

    // 发布更新后的路径
    pub_pl_func(pcl_path, pub_curr_path);

    // 更新当前位姿
    x_curr.R = x_buf[win_count - 1].R;
    x_curr.p = x_buf[win_count - 1].p;
    x_curr.v = dx.R * x_curr.v;
    x_curr.g = x_buf[win_count - 1].g;

    // 重置窗口索引
    for (int i = 0; i < win_size; i++)
      mp[i] = i;

    // 更新回环检测到的特征点
    for (ScanPose *bl : buf_lba2loop)
    {
      IMUST xx = bl->x;
      PVec pvec_tem = *(bl->pvec);
      // 将特征点转换到世界坐标系
      for (pointVar &pv : pvec_tem)
        pv.pnt = xx.R * pv.pnt + xx.p;
      // 将特征点添加到体素地图
      cut_voxel(surf_map, pvec_tem, win_size, 0);
    }

    // 更新当前窗口内的特征点
    PLV(3) pwld; // 世界坐标系下的点坐标容器
    for (int i = 0; i < win_count; i++)
    {
      pwld.clear();
      // 将特征点转换到世界坐标系
      for (pointVar &pv : *pvec_buf[i])
        pwld.push_back(x_buf[i].R * pv.pnt + x_buf[i].p);
      // 将特征点添加到体素地图
      cut_voxel(surf_map, pvec_buf[i], i, surf_map_slide, win_size, pwld, sws[0]);
    }

    // 重新切分所有特征
    for (auto iter = surf_map.begin(); iter != surf_map.end(); ++iter)
      iter->second->recut(win_count, x_buf, sws[0]);

    // 更新重力向量状态
    if (g_update == 1)
      g_update = 2;
    // 重置回环检测标志
    loop_detect = 0;
    // 打印处理时间
    double t2 = ros::Time::now().toSec();
    printf("loop head: %lf %zu\n", t2 - t1, sws[0].size());
  }

  /**
   * @brief 将之前的关键帧加载到局部体素地图中
   * @param jour 当前时间戳
   */
  void keyframe_loading(double jour)
  {
    // 如果没有历史关键帧，直接返回
    if (history_kfsize <= 0)
      return;
    double tt1 = ros::Time::now().toSec();

    // 构造当前位置的点用于KD树搜索
    PointType ap_curr;
    ap_curr.x = x_curr.p[0];
    ap_curr.y = x_curr.p[1];
    ap_curr.z = x_curr.p[2];

    // 用于存储KD树搜索结果
    vector<int> vec_idx;
    vector<float> vec_dis;
    // 在KD树中搜索10米范围内的关键帧
    kd_keyframes.radiusSearch(ap_curr, 10, vec_idx, vec_dis);

    // 遍历搜索到的关键帧
    for (int id : vec_idx)
    {
      // 获取关键帧的序号
      int ord_kf = pl_kdmap->points[id].curvature;
      // 检查关键帧是否有效
      if (keyframes->at(id)->exist)
      {
        // 获取关键帧引用
        Keyframe &kf = *(keyframes->at(id));
        IMUST &xx = kf.x0;  // 获取关键帧位姿
        PVec pvec;  // 用于存储转换后的点云
        pvec.reserve(kf.plptr->size());

        // 初始化点的方差
        pointVar pv;
        pv.var.setZero();
        int plsize = kf.plptr->size();
        
        // 遍历关键帧中的所有点
        for (int j = 0; j < plsize; j++)
        {
          // 获取原始点坐标
          PointType ap = kf.plptr->points[j];
          // 将点坐标转换为向量形式
          pv.pnt << ap.x, ap.y, ap.z;
          // 将点从关键帧坐标系转换到世界坐标系
          pv.pnt = xx.R * pv.pnt + xx.p;
          pvec.push_back(pv);
        }

        // 将转换后的点云添加到体素地图中
        cut_voxel(surf_map, pvec, win_size, jour);
        // 标记该关键帧已被处理
        kf.exist = 0;
        // 更新历史关键帧计数
        history_kfsize--;
        break;
      }
    }
  }

  /**
   * 初始化SLAM系统
   *
   * @param imus IMU数据队列，包含角速度和加速度测量值
   * @param hess Hessian矩阵，用于优化过程
   * @param voxhess 体素因子，用于处理体素地图
   * @param pwld 世界坐标系中的点云向量
   * @param pcl_curr 当前帧点云数据
   * @return 初始化状态：0-继续收集数据，1-初始化成功，-1-初始化失败
   */
  int initialization(deque<sensor_msgs::Imu::Ptr> &imus, Eigen::MatrixXd &hess, LidarFactor &voxhess, PLV(3) & pwld, pcl::PointCloud<PointType>::Ptr pcl_curr)
  {
    // 静态变量存储多帧原始点云、起始时间和IMU数据
    static vector<pcl::PointCloud<PointType>::Ptr> pl_origs;
    static vector<double> beg_times;
    static vector<deque<sensor_msgs::Imu::Ptr>> vec_imus;

    // 复制当前点云数据
    pcl::PointCloud<PointType>::Ptr orig(new pcl::PointCloud<PointType>(*pcl_curr));
    // 使用EKF处理IMU和点云数据，失败则返回0
    if (odom_ekf.process(x_curr, *pcl_curr, imus) == 0)
      return 0;

    // 第一次初始化时记录重力尺度
    if (win_count == 0)
      imupre_scale_gravity = odom_ekf.scale_gravity;

    // 创建点云向量指针
    PVecPtr pptr(new PVec);
    // 设置下采样尺寸
    double downkd = down_size >= 0.5 ? down_size : 0.5;
    // 进行点云下采样
    down_sampling_voxel(*pcl_curr, downkd);
    // 初始化点云协方差
    var_init(extrin_para, *pcl_curr, pptr, dept_err, beam_err);
    // 使用k-d树进行点到平面的配准
    lio_state_estimation_kdtree(pptr);

    // 清除并更新世界坐标系中的点云
    pwld.clear();
    pvec_update(pptr, x_curr, pwld);

    // 窗口计数增加
    win_count++;
    // 保存当前状态和点云
    x_buf.push_back(x_curr);
    pvec_buf.push_back(pptr);
    // 发布局部轨迹
    ResultOutput::instance().pub_localtraj(pwld, 0, x_curr, sessionNames.size() - 1, pcl_path);

    // 如果不是第一帧，处理IMU数据
    if (win_count > 1)
    {
      // 创建IMU预积分并添加IMU数据
      imu_pre_buf.push_back(new IMU_PRE(x_buf[win_count - 2].bg, x_buf[win_count - 2].ba));
      imu_pre_buf[win_count - 2]->push_imu(imus);
    }

    // 保存中间点云
    pcl::PointCloud<PointType> pl_mid = *orig;
    // 下采样原始点云
    down_sampling_close(*orig, down_size);
    // 如果下采样后点数太少，使用较小的下采样参数重新采样
    if (orig->size() < 1000)
    {
      *orig = pl_mid;
      down_sampling_close(*orig, down_size / 2);
    }

    // 根据曲率排序点云
    sort(orig->begin(), orig->end(), [](PointType &x, PointType &y)
         { return x.curvature < y.curvature; });

    // 保存原始点云、起始时间和IMU数据
    pl_origs.push_back(orig);
    beg_times.push_back(odom_ekf.pcl_beg_time);
    vec_imus.push_back(imus);

    // 初始化状态
    int is_success = 0;
    // 当收集了足够的帧后执行运动初始化
    if (win_count >= win_size)
    {
      // 执行运动初始化
      is_success = Initialization::instance().motion_init(pl_origs, vec_imus, beg_times, &hess, voxhess, x_buf, surf_map, surf_map_slide, pvec_buf, win_size, sws, x_curr, imu_pre_buf, extrin_para);

      // 初始化失败返回-1
      if (is_success == 0)
        return -1;
      // 初始化成功返回1
      return 1;
    }
    // 继续收集数据
    return 0;
  }

  // 系统重置函数，用于重置SLAM系统的状态
  void system_reset(deque<sensor_msgs::Imu::Ptr> &imus)
  {
    // 清理表面特征地图
    for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
    {
      // 将八叉树节点转移到释放队列
      iter->second->tras_ptr(octos_release);
      // 清除滑动窗口中的特征
      iter->second->clear_slwd(sws[0]);
      // 删除八叉树节点
      delete iter->second;
    }
    // 清空表面特征地图和滑动窗口特征地图
    surf_map.clear();
    surf_map_slide.clear();

    // 重置当前位姿状态
    x_curr.setZero();
    // 设置初始位置为(0,0,30)
    x_curr.p = Eigen::Vector3d(0, 0, 30);
    // 重置IMU EKF状态
    odom_ekf.mean_acc.setZero();
    odom_ekf.init_num = 0;
    // 重新初始化IMU
    odom_ekf.IMU_init(imus);
    // 设置重力向量
    x_curr.g = -odom_ekf.mean_acc * imupre_scale_gravity;

    // 清理IMU预积分缓冲区
    for (int i = 0; i < imu_pre_buf.size(); i++)
      delete imu_pre_buf[i];
    // 清空各种缓冲区
    x_buf.clear();
    pvec_buf.clear();
    imu_pre_buf.clear();
    pl_tree->clear();

    // 重置滑动窗口索引
    for (int i = 0; i < win_size; i++)
      mp[i] = i;
    // 重置窗口基址和计数
    win_base = 0;
    win_count = 0;
    // 清空点云路径
    pcl_path.clear();
    // 发布清空后的地图
    pub_pl_func(pcl_path, pub_cmap);
    // 输出重置提示
    ROS_WARN("Reset");
  }

  // After local BA, update the map and marginalize the points of oldest scan
  // multi means multiple thread
  void multi_margi(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, double jour, int win_count, vector<IMUST> &xs, LidarFactor &voxopt, vector<SlideWindow *> &sw)
  {
    // 注释掉的代码是单线程版本，用于参考
    // for(auto iter=feat_map.begin(); iter!=feat_map.end();)
    // {
    //   iter->second->jour = jour;
    //   iter->second->margi(win_count, 1, xs, voxopt);
    //   if(iter->second->isexist)
    //     iter++;
    //   else
    //   {
    //     iter->second->clear_slwd(sw);
    //     feat_map.erase(iter++);
    //   }
    // }
    // return;

    // 设置线程数量
    int thd_num = thread_num;
    // 为每个线程创建八叉树节点容器
    vector<vector<OctoTree *> *> octs;
    for (int i = 0; i < thd_num; i++)
      octs.push_back(new vector<OctoTree *>());

    // 获取特征地图大小，如果小于线程数则直接返回
    int g_size = feat_map.size();
    if (g_size < thd_num)
      return;

    // 创建线程数组
    vector<thread *> mthreads(thd_num);
    // 计算每个线程需要处理的特征数量
    double part = 1.0 * g_size / thd_num;
    int cnt = 0;

    // 将特征均匀分配到各个线程的容器中
    for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++)
    {
      iter->second->jour = jour;
      octs[cnt]->push_back(iter->second);
      if (octs[cnt]->size() >= part && cnt < thd_num - 1)
        cnt++;
    }

    // 定义边缘化处理函数
    auto margi_func = [](int win_cnt, vector<OctoTree *> *oct, vector<IMUST> xxs, LidarFactor &voxhess)
    {
      // 对每个八叉树节点执行边缘化操作
      for (OctoTree *oc : *oct)
      {
        oc->margi(win_cnt, 1, xxs, voxhess);
      }
    };

    // 启动其他线程执行边缘化
    for (int i = 1; i < thd_num; i++)
    {
      mthreads[i] = new thread(margi_func, win_count, octs[i], xs, ref(voxopt));
    }

    // 主线程执行第一个分区的边缘化，并等待其他线程完成
    for (int i = 0; i < thd_num; i++)
    {
      if (i == 0)
      {
        margi_func(win_count, octs[i], xs, voxopt);
      }
      else
      {
        mthreads[i]->join();
        delete mthreads[i];
      }
    }

    // 清理不存在的特征
    for (auto iter = feat_map.begin(); iter != feat_map.end();)
    {
      if (iter->second->isexist)
        iter++;
      else
      {
        iter->second->clear_slwd(sw);
        feat_map.erase(iter++);
      }
    }

    // 释放线程容器内存
    for (int i = 0; i < thd_num; i++)
      delete octs[i];
  }

  // Determine the plane and recut the voxel map in octo-tree
  void multi_recut(unordered_map<VOXEL_LOC, OctoTree *> &feat_map, int win_count, vector<IMUST> &xs, LidarFactor &voxopt, vector<vector<SlideWindow *>> &sws)
  {
    // for(auto iter=feat_map.begin(); iter!=feat_map.end(); iter++)
    // {
    //   iter->second->recut(win_count, xs, sws[0]);
    //   iter->second->tras_opt(voxopt);
    // }

    int thd_num = thread_num;
    vector<vector<OctoTree *>> octss(thd_num);
    int g_size = feat_map.size();
    if (g_size < thd_num)
      return;
    vector<thread *> mthreads(thd_num);
    double part = 1.0 * g_size / thd_num;
    int cnt = 0;
    for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++)
    {
      octss[cnt].push_back(iter->second);
      if (octss[cnt].size() >= part && cnt < thd_num - 1)
        cnt++;
    }

    auto recut_func = [](int win_count, vector<OctoTree *> &oct, vector<IMUST> xxs, vector<SlideWindow *> &sw)
    {
      for (OctoTree *oc : oct)
        oc->recut(win_count, xxs, sw);
    };

    for (int i = 1; i < thd_num; i++)
    {
      mthreads[i] = new thread(recut_func, win_count, ref(octss[i]), xs, ref(sws[i]));
    }

    for (int i = 0; i < thd_num; i++)
    {
      if (i == 0)
      {
        recut_func(win_count, octss[i], xs, sws[i]);
      }
      else
      {
        mthreads[i]->join();
        delete mthreads[i];
      }
    }

    for (int i = 1; i < sws.size(); i++)
    {
      sws[0].insert(sws[0].end(), sws[i].begin(), sws[i].end());
      sws[i].clear();
    }

    for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++)
      iter->second->tras_opt(voxopt);
  }

  // 里程计和局部建图线程函数
  void thd_odometry_localmapping(ros::NodeHandle &n)
  {
    // 初始化变量
    PLV(3) pwld;  // 世界坐标系下的点云
    double down_sizes[3] = {0.1, 0.2, 0.4};  // 降采样体素大小
    Eigen::Vector3d last_pos(0, 0, 0);  // 上一帧位置
    double jour = 0;  // 时间戳
    int counter = 0;  // 计数器

    // 点云相关变量初始化
    pcl::PointCloud<PointType>::Ptr pcl_curr(new pcl::PointCloud<PointType>());
    int motion_init_flag = 1;  // 运动初始化标志
    pl_tree.reset(new pcl::PointCloud<PointType>());
    vector<pcl::PointCloud<PointType>::Ptr> pl_origs;  // 原始点云
    vector<double> beg_times;  // 开始时间
    vector<deque<sensor_msgs::Imu::Ptr>> vec_imus;  // IMU数据队列
    bool release_flag = false;  // 释放标志
    int degrade_cnt = 0;  // 退化计数
    LidarFactor voxhess(win_size);  // 激光雷达因子
    const int mgsize = 1;  // 边缘化大小
    Eigen::MatrixXd hess;  // 海森矩阵

    // 主循环
    while (n.ok())
    {
      ros::spinOnce();  // 处理ROS回调

      // 处理回环检测
      if (loop_detect == 1)
      {
        loop_update();
        last_pos = x_curr.p;
        jour = 0;
      }

      // 检查是否结束
      n.param<bool>("finish", is_finish, false);
      if (is_finish)
      {
        break;
      }

      // 同步数据包
      deque<sensor_msgs::Imu::Ptr> imus;
      if (!sync_packages(pcl_curr, imus, odom_ekf))
      {
        // 内存管理：释放八叉树节点
        if (octos_release.size() != 0)
        {
          int msize = octos_release.size();
          if (msize > 1000)
            msize = 1000;
          for (int i = 0; i < msize; i++)
          {
            delete octos_release.back();
            octos_release.pop_back();
          }
          malloc_trim(0);
        }
        // 释放长时间未使用的特征
        else if (release_flag)
        {
          release_flag = false;
          vector<OctoTree *> octos;
          for (auto iter = surf_map.begin(); iter != surf_map.end();)
          {
            int dis = jour - iter->second->jour;
            if (dis < 700)
            {
              iter++;
            }
            else
            {
              octos.push_back(iter->second);
              iter->second->tras_ptr(octos);
              surf_map.erase(iter++);
            }
          }
          int ocsize = octos.size();
          for (int i = 0; i < ocsize; i++)
            delete octos[i];
          octos.clear();
          malloc_trim(0);
        }
        // 释放滑动窗口
        else if (sws[0].size() > 10000)
        {
          for (int i = 0; i < 500; i++)
          {
            delete sws[0].back();
            sws[0].pop_back();
          }
          malloc_trim(0);
        }

        sleep(0.001);
        continue;
      }

      // 第一次运行时的初始化
      static int first_flag = 1;
      if (first_flag)
      {
        pcl::PointCloud<PointType> pl;
        pub_pl_func(pl, pub_pmap);
        pub_pl_func(pl, pub_prev_path);
        first_flag = 0;
      }

      // 记录时间点
      double t0 = ros::Time::now().toSec();
      double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0;

      // 运动初始化
      if (motion_init_flag)
      {
        int init = initialization(imus, hess, voxhess, pwld, pcl_curr);

        if (init == 1)
        {
          motion_init_flag = 0;
        }
        else
        {
          if (init == -1)
            system_reset(imus);
          continue;
        }
      }
      else
      {
        // IMU里程计处理
        if (odom_ekf.process(x_curr, *pcl_curr, imus) == 0)
          continue;

        // 点云降采样
        pcl::PointCloud<PointType> pl_down = *pcl_curr;
        down_sampling_voxel(pl_down, down_size);

        if (pl_down.size() < 500)
        {
          pl_down = *pcl_curr;
          down_sampling_voxel(pl_down, down_size / 2);
        }

        // 初始化点云向量
        PVecPtr pptr(new PVec);
        var_init(extrin_para, pl_down, pptr, dept_err, beam_err);

        // 激光雷达状态估计
        if (lio_state_estimation(pptr))
        {
          if (degrade_cnt > 0)
            degrade_cnt--;
        }
        else
          degrade_cnt++;

        // 更新世界坐标系下的点云
        pwld.clear();
        pvec_update(pptr, x_curr, pwld);
        ResultOutput::instance().pub_localtraj(pwld, jour, x_curr, sessionNames.size() - 1, pcl_path);

        t1 = ros::Time::now().toSec();

        // 更新窗口状态
        win_count++;
        x_buf.push_back(x_curr);
        pvec_buf.push_back(pptr);
        if (win_count > 1)
        {
          imu_pre_buf.push_back(new IMU_PRE(x_buf[win_count - 2].bg, x_buf[win_count - 2].ba));
          imu_pre_buf[win_count - 2]->push_imu(imus);
        }

        // 加载关键帧
        keyframe_loading(jour);
        voxhess.clear();
        voxhess.win_size = win_size;

        // 体素化处理
        cut_voxel_multi(surf_map, pvec_buf[win_count - 1], win_count - 1, surf_map_slide, win_size, pwld, sws);
        t2 = ros::Time::now().toSec();

        // 多线程重切分
        multi_recut(surf_map_slide, win_count, x_buf, voxhess, sws);
        t3 = ros::Time::now().toSec();

        // 处理退化情况
        if (degrade_cnt > degrade_bound)
        {
          degrade_cnt = 0;
          system_reset(imus);

          last_pos = x_curr.p;
          jour = 0;

          mtx_loop.lock();
          buf_lba2loop_tem.swap(buf_lba2loop);
          mtx_loop.unlock();
          reset_flag = 1;

          motion_init_flag = 1;
          history_kfsize = 0;

          continue;
        }
      }

      // 窗口满时的处理
      if (win_count >= win_size)
      {
        t4 = ros::Time::now().toSec();

        // 重力更新
        if (g_update == 2)
        {
          LI_BA_OptimizerGravity opt_lsv;
          vector<double> resis;
          opt_lsv.damping_iter(x_buf, voxhess, imu_pre_buf, resis, &hess, 5);
          printf("g update: %lf %lf %lf: %lf\n", x_buf[0].g[0], x_buf[0].g[1], x_buf[0].g[2], x_buf[0].g.norm());
          g_update = 0;
          x_curr.g = x_buf[win_count - 1].g;
        }
        else
        {
          // 局部BA优化
          LI_BA_Optimizer opt_lsv;
          opt_lsv.damping_iter(x_buf, voxhess, imu_pre_buf, &hess);
        }

        // 创建新的扫描位姿
        ScanPose *bl = new ScanPose(x_buf[0], pvec_buf[0]);
        bl->v6 = hess.block<6, 6>(0, DIM).diagonal();
        for (int i = 0; i < 6; i++)
          bl->v6[i] = 1.0 / fabs(bl->v6[i]);
        mtx_loop.lock();
        buf_lba2loop.push_back(bl);
        mtx_loop.unlock();

        // 更新当前位姿
        x_curr.R = x_buf[win_count - 1].R;
        x_curr.p = x_buf[win_count - 1].p;
        t5 = ros::Time::now().toSec();

        // 发布局部地图
        ResultOutput::instance().pub_localmap(mgsize, sessionNames.size() - 1, pvec_buf, x_buf, pcl_path, win_base, win_count);

        // 多线程边缘化
        multi_margi(surf_map_slide, jour, win_count, x_buf, voxhess, sws[0]);
        t6 = ros::Time::now().toSec();

        // 更新jour和last_pos
        if ((win_base + win_count) % 10 == 0)
        {
          double spat = (x_curr.p - last_pos).norm();
          if (spat > 0.5)
          {
            jour += spat;
            last_pos = x_curr.p;
            release_flag = true;
          }
        }

        // 保存地图
        if (is_save_map)
        {
          for (int i = 0; i < mgsize; i++)
            FileReaderWriter::instance().save_pcd(pvec_buf[i], x_buf[i], win_base + i, savepath + bagname);
        }

        // 更新窗口索引
        for (int i = 0; i < win_size; i++)
        {
          mp[i] += mgsize;
          if (mp[i] >= win_size)
            mp[i] -= win_size;
        }

        // 滑动窗口更新
        for (int i = mgsize; i < win_count; i++)
        {
          x_buf[i - mgsize] = x_buf[i];
          PVecPtr pvec_tem = pvec_buf[i - mgsize];
          pvec_buf[i - mgsize] = pvec_buf[i];
          pvec_buf[i] = pvec_tem;
        }

        // 删除旧数据
        for (int i = win_count - mgsize; i < win_count; i++)
        {
          x_buf.pop_back();
          pvec_buf.pop_back();

          delete imu_pre_buf.front();
          imu_pre_buf.pop_front();
        }

        // 更新窗口参数
        win_base += mgsize;
        win_count -= mgsize;
      }

      // 记录时间和内存使用
      double t_end = ros::Time::now().toSec();
      double mem = get_memory();
    }

    // 清理内存
    vector<OctoTree *> octos;
    for (auto iter = surf_map.begin(); iter != surf_map.end(); iter++)
    {
      iter->second->tras_ptr(octos);
      iter->second->clear_slwd(sws[0]);
      delete iter->second;
    }

    for (int i = 0; i < octos.size(); i++)
      delete octos[i];
    octos.clear();

    for (int i = 0; i < sws[0].size(); i++)
      delete sws[0][i];
    sws[0].clear();
    malloc_trim(0);
  }

  // 构建位姿图用于回环检测
  // initial: 初始位姿值
  // graph: 非线性因子图
  // cur_id: 当前地图ID
  // lp_edges: 回环边集合
  // default_noise: 默认噪声模型
  // ids: 地图ID列表
  // stepsizes: 每个地图的位姿数量累加和
  // lpedge_enable: 是否启用回环边
  void build_graph(gtsam::Values &initial, gtsam::NonlinearFactorGraph &graph, int cur_id, PGO_Edges &lp_edges, gtsam::noiseModel::Diagonal::shared_ptr default_noise, vector<int> &ids, vector<int> &stepsizes, int lpedge_enable)
  {
    // 清空初始值和因子图
    initial.clear();
    graph = gtsam::NonlinearFactorGraph();
    ids.clear();
    // 连接当前地图ID到其他地图
    lp_edges.connect(cur_id, ids);

    // 计算每个地图的位姿数量累加和
    stepsizes.clear();
    stepsizes.push_back(0);
    for (int i = 0; i < ids.size(); i++)
      stepsizes.push_back(stepsizes.back() + multimap_scanPoses[ids[i]]->size());

    // 遍历所有地图
    for (int ii = 0; ii < ids.size(); ii++)
    {
      int bsize = stepsizes[ii], id = ids[ii];
      // 遍历当前地图的所有位姿
      for (int j = bsize; j < stepsizes[ii + 1]; j++)
      {
        // 获取当前位姿
        IMUST &xc = multimap_scanPoses[id]->at(j - bsize)->x;
        // 将位姿转换为gtsam格式并插入初始值
        gtsam::Pose3 pose3(gtsam::Rot3(xc.R), gtsam::Point3(xc.p));
        initial.insert(j, pose3);
        // 如果不是第一个位姿，添加相邻位姿间的边
        if (j > bsize)
        {
          // 使用位姿的协方差作为噪声模型
          gtsam::Vector samv6(6);
          samv6 = multimap_scanPoses[ids[ii]]->at(j - 1 - bsize)->v6;
          gtsam::noiseModel::Diagonal::shared_ptr v6_noise = gtsam::noiseModel::Diagonal::Variances(samv6);
          add_edge(j - 1, j, multimap_scanPoses[id]->at(j - 1 - bsize)->x, multimap_scanPoses[id]->at(j - bsize)->x, graph, v6_noise);
        }
      }
    }

    // 为第一个地图添加先验约束
    if (multimap_scanPoses[ids[0]]->size() != 0)
    {
      int ceil = multimap_scanPoses[ids[0]]->size();
      ceil = 1; // 只固定第一个位姿
      for (int i = 0; i < ceil; i++)
      {
        // 使用很小的方差作为先验约束的噪声模型
        Eigen::Matrix<double, 6, 1> v6_fixd;
        v6_fixd << 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9;
        gtsam::noiseModel::Diagonal::shared_ptr fixd_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector(v6_fixd));
        // 获取固定位姿并添加到因子图中
        IMUST xf = multimap_scanPoses[ids[0]]->at(i)->x;
        gtsam::Pose3 pose3 = gtsam::Pose3(gtsam::Rot3(xf.R), gtsam::Point3(xf.p));
        // graph.addPrior(i, pose3, fixd_noise);
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(i, pose3, fixd_noise));
      }
    }

    // 如果启用回环边，添加回环约束
    if (lpedge_enable == 1)
      for (PGO_Edge &edge : lp_edges.edges)
      {
        vector<int> step(2);
        // 检查回环边是否适用于当前地图集合
        if (edge.is_adapt(ids, step))
        {
          int mp[2] = {stepsizes[step[0]], stepsizes[step[1]]};
          // 添加所有回环边
          for (int i = 0; i < edge.rots.size(); i++)
          {
            int id1 = mp[0] + edge.ids1[i];
            int id2 = mp[1] + edge.ids2[i];
            add_edge(id1, id2, edge.rots[i], edge.tras[i], graph, default_noise);
          }
        }
      }
  }

  // The main thread of loop clousre
  // The topDownProcess of HBA is also run here
  void thd_loop_closure(ros::NodeHandle &n)
  {
    // 初始化关键帧地图点云
    pl_kdmap.reset(new pcl::PointCloud<PointType>);
    vector<STDescManager *> std_managers;  // ST描述子管理器列表
    PGO_Edges lp_edges;  // 回环边集合

    // 初始化回环检测参数
    double jud_default = 0.45, icp_eigval = 14;  // 回环判断阈值和ICP特征值阈值
    double ratio_drift = 0.05;  // 漂移比例阈值
    int curr_halt = 10, prev_halt = 30;  // 当前和之前会话的暂停阈值
    int isHighFly = 0;  // 是否高空飞行标志
    // 从ROS参数服务器读取回环检测参数
    n.param<double>("Loop/jud_default", jud_default, 0.45);
    n.param<double>("Loop/icp_eigval", icp_eigval, 14);
    n.param<double>("Loop/ratio_drift", ratio_drift, 0.05);
    n.param<int>("Loop/curr_halt", curr_halt, 10);
    n.param<int>("Loop/prev_halt", prev_halt, 30);
    n.param<int>("Loop/isHighFly", isHighFly, 0);
    ConfigSetting config_setting;  // 配置设置
    read_parameters(n, config_setting, isHighFly);

    // 读取之前的地图会话信息
    vector<double> juds;
    FileReaderWriter::instance().previous_map_names(n, sessionNames, juds);
    FileReaderWriter::instance().pgo_edges_io(lp_edges, sessionNames, 0, savepath, bagname);
    FileReaderWriter::instance().previous_map_read(std_managers, multimap_scanPoses, multimap_keyframes, config_setting, lp_edges, n, sessionNames, juds, savepath, win_size);

    // 初始化当前会话的ST描述子管理器
    STDescManager *std_manager = new STDescManager(config_setting);
    sessionNames.push_back(bagname);
    std_managers.push_back(std_manager);
    multimap_scanPoses.push_back(scanPoses);
    multimap_keyframes.push_back(keyframes);
    juds.push_back(jud_default);
    vector<double> jours(std_managers.size(), 0);  // 记录每个会话的里程

    // 初始化回环计数
    vector<int> relc_counts(std_managers.size(), prev_halt);

    // 初始化局部缓冲区
    deque<ScanPose *> bl_local;
    // 初始化位姿图优化参数
    Eigen::Matrix<double, 6, 1> v6_init, v6_fixd;
    v6_init << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;  // 初始协方差
    v6_fixd << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;  // 固定协方差
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector(v6_init));
    gtsam::noiseModel::Diagonal::shared_ptr fixd_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector(v6_fixd));
    gtsam::Values initial;  // 初始位姿值
    gtsam::NonlinearFactorGraph graph;  // 位姿图

    // 初始化会话ID和步长
    vector<int> ids(1, std_managers.size() - 1), stepsizes(2, 0);
    pcl::PointCloud<pcl::PointXYZI>::Ptr plbtc(new pcl::PointCloud<pcl::PointXYZI>);
    IMUST x_key;  // 关键帧位姿
    int buf_base = 0;  // 缓冲区基准索引

    // 主循环
    while (n.ok())
    {
      // 处理系统重置
      if (reset_flag == 1)
      {
        reset_flag = 0;

        // 将临时缓冲区的位姿添加到历史位姿列表
        scanPoses->insert(scanPoses->end(), buf_lba2loop_tem.begin(), buf_lba2loop_tem.end());

        // 清理点云指针防止内存泄漏
        for (ScanPose *bl : buf_lba2loop_tem)
          bl->pvec = nullptr;
        buf_lba2loop_tem.clear();

        // 创建新的地图容器
        keyframes = new vector<Keyframe *>(); // 新的关键帧列表
        multimap_keyframes.push_back(keyframes); // 添加到多地图关键帧数组
        scanPoses = new vector<ScanPose *>(); // 新的扫描位姿列表
        multimap_scanPoses.push_back(scanPoses); // 添加到多地图位姿数组

        // 重置相关变量
        bl_local.clear(); // 清空局部缓冲区
        buf_base = 0; // 重置缓冲区基础索引

        // 重置描述子管理器，避免与先前地图产生错误关联
        std_manager->config_setting_.skip_near_num_ = -(std_manager->plane_cloud_vec_.size() + 10);
        std_manager = new STDescManager(config_setting); // 创建新的描述子管理器
        std_managers.push_back(std_manager); // 添加到管理器列表

        // 记录各种统计数据
        relc_counts.push_back(prev_halt); // 回环计数
        sessionNames.push_back(bagname + to_string(sessionNames.size())); // 会话名称
        juds.push_back(jud_default); // 回环判断阈值
        jours.push_back(0); // 里程

        // 更新当前包名
        bagname = sessionNames.back();
        string cmd = "mkdir " + savepath + bagname + "/";
        int ss = system(cmd.c_str());

        // 发布全局路径和地图可视化
        ResultOutput::instance().pub_global_path(multimap_scanPoses, pub_prev_path, ids);
        ResultOutput::instance().pub_globalmap(multimap_keyframes, ids, pub_pmap);

        // 重置位姿图优化相关数据
        initial.clear(); // 清空初始值
        graph = gtsam::NonlinearFactorGraph(); // 创建新的因子图
        ids.clear(); // 清空地图ID列表
        ids.push_back(std_managers.size() - 1); // 添加新地图ID
        stepsizes.clear(); // 清空步长列表
        stepsizes.push_back(0); // 添加起始步长
        stepsizes.push_back(0); // 添加结束步长
      }

      // 检查是否完成且缓冲区为空
      if (is_finish && buf_lba2loop.empty())
      {
        break;
      }

      // 如果缓冲区为空或正在检测回环，则等待
      if (buf_lba2loop.empty() || loop_detect == 1)
      {
        sleep(0.01);
        continue;
      }

      // 从缓冲区获取新的扫描位姿
      ScanPose *bl_head = nullptr;
      mtx_loop.lock();
      if (!buf_lba2loop.empty())
      {
        bl_head = buf_lba2loop.front(); // 获取缓冲区头部元素
        buf_lba2loop.pop_front();
      }
      mtx_loop.unlock();
      if (bl_head == nullptr)
        continue;

      // 处理新的扫描位姿
      int cur_id = std_managers.size() - 1; // 当前地图ID
      scanPoses->push_back(bl_head); // 添加到扫描位姿列表
      bl_local.push_back(bl_head); // 添加到局部缓冲区
      IMUST xc = bl_head->x; // 获取当前扫描位姿

      // 转换为GTSAM位姿格式
      gtsam::Pose3 pose3(gtsam::Rot3(xc.R), gtsam::Point3(xc.p));
      int g_pos = stepsizes.back(); // 获取当前位置在因子图中的索引
      initial.insert(g_pos, pose3); // 添加到初始值

      // 添加位姿图边
      if (g_pos > 0) // 如果不是第一个位姿
      {
        // 创建噪声模型，使用前一个位姿的不确定性
        gtsam::Vector samv6(scanPoses->at(buf_base - 1)->v6);
        gtsam::noiseModel::Diagonal::shared_ptr v6_noise = gtsam::noiseModel::Diagonal::Variances(samv6);

        // 添加相邻位姿之间的边（里程计约束）
        add_edge(g_pos - 1, g_pos, scanPoses->at(buf_base - 1)->x, xc, graph, v6_noise);
      }
      else // 如果是第一个位姿
      {
        // 添加先验约束固定第一个位姿
        gtsam::Pose3 pose3(gtsam::Rot3(xc.R), gtsam::Point3(xc.p));
        // graph.addPrior(0, pose3, fixd_noise);
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, pose3, fixd_noise));
      }

      // 更新关键帧信息
      if (buf_base == 0) // 如果是第一个位姿，保存位姿作为关键帧
        x_key = xc;
      buf_base++; // 增加缓冲区基础索引
      stepsizes.back() += 1; // 增加步长

      // 检查是否需要创建新的关键帧
      if (bl_local.size() < win_size)
        continue;

      // 计算当前位姿与关键位姿之间的变化
      double ang = Log(x_key.R.transpose() * xc.R).norm() * 57.3;
      double len = (xc.p - x_key.p).norm();

      // 如果变化不够大，丢弃最早的位姿，不创建关键帧
      if (ang < 5 && len < 0.1 && buf_base > win_size)
      {
        bl_local.front()->pvec = nullptr;
        bl_local.pop_front();
        continue;
      }

      // 更新各地图的累积路程
      for (double &jour : jours)
        jour += len;

      // 更新关键位姿
      x_key = xc;

      // 处理点云数据，合并局部窗口内的点云
      PVecPtr pptr(new PVec); // 创建新的点云容器
      // 遍历局部窗口
      for (int i = 0; i < win_size; i++)
      {
        ScanPose &bl = *bl_local[i]; // 获取位姿

        // 计算当前帧与局部窗口内的相对位姿变换
        Eigen::Vector3d delta_p = xc.R.transpose() * (bl.x.p - xc.p);
        Eigen::Matrix3d delta_R = xc.R.transpose() * bl.x.R;

        // 转换点云到当前位姿坐标系
        for (pointVar pv : *(bl.pvec))
        {
          pv.pnt = delta_R * pv.pnt + delta_p;
          pptr->push_back(pv);
        }
      }
      // 清空局部窗口
      for (int i = 0; i < win_size; i++)
      {
        bl_local.front()->pvec = nullptr;
        bl_local.pop_front();
      }

      // 创建新的关键帧
      Keyframe *smp = new Keyframe(xc); // 创建关键帧对象
      smp->id = buf_base - 1; // 设置ID
      smp->jour = jours[cur_id]; // 设置累积路程

      // 下采样点云并存储到关键帧
      down_sampling_pvec(*pptr, voxel_size / 10, *(smp->plptr));

      // 转换点云格式用于描述子生成
      plbtc->clear(); // 清空临时点云
      pcl::PointXYZI ap;
      for (pointVar &pv : *pptr) // 遍历合并点云
      {
        Eigen::Vector3d &wld = pv.pnt;
        ap.x = wld[0];
        ap.y = wld[1];
        ap.z = wld[2];
        plbtc->push_back(ap);
      }

      // 添加关键帧到列表
      mtx_keyframe.lock();
      keyframes->push_back(smp);
      mtx_keyframe.unlock();

      // 生成ST描述子并进行回环检测
      vector<STD> stds_vec; // 描述子向量
      std_manager->GenerateSTDescs(plbtc, stds_vec, buf_base - 1);

      // 初始化回环检测结果变量
      pair<int, double> search_result(-1, 0); // 搜索结果<索引,得分>
      pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform; // 回环变换<平移,旋转>
      vector<pair<STD, STD>> loop_std_pair; // 匹配的描述子对

      // 回环检测和优化标志
      bool isGraph = false, isOpt = false; // 是否需要构建图/优化
      int match_num = 0; // 匹配数量

      // 对所有地图进行回环检测
      for (int id = 0; id <= cur_id; id++)
      {
        // 搜索回环匹配
        std_managers[id]->SearchLoop(stds_vec, search_result, loop_transform, loop_std_pair, std_manager->plane_cloud_vec_.back());

        // 如果找到匹配，打印信息
        if (search_result.first >= 0)
        {
          printf("Find Loop in session%d: %d %d\n", id, buf_base, search_result.first);
          printf("score: %lf\n", search_result.second);
        }

        // 如果找到足够好的匹配
        if (search_result.first >= 0 && search_result.second > juds[id])
        {
          // 使用ICP细化变换
          if (icp_normal(*(std_manager->plane_cloud_vec_.back()), *(std_managers[id]->plane_cloud_vec_[search_result.first]), loop_transform, icp_eigval))
          {
            // 获取匹配到的历史扫描位姿索引
            int ord_bl = std_managers[id]->plane_cloud_vec_[search_result.first]->header.seq;

            // 计算位姿漂移
            IMUST &xx = multimap_scanPoses[id]->at(ord_bl)->x;
            double drift_p = (xx.R * loop_transform.first + xx.p - xc.p).norm();

            // 处理回环变量
            bool isPush = false; // 是否添加约束
            int step = -1; // 地图索引

            // 如果是当前地图内的回环
            if (id == cur_id)
            {
              double span = smp->jour - keyframes->at(search_result.first)->jour; // 路程差
              printf("drift: %lf %lf\n", drift_p, span); // 打印漂移和路程

              // 如果漂移率低于阈值
              if (drift_p / span < ratio_drift)
              {
                isPush = true; // 标记添加约束
                step = stepsizes.size() - 2; // 设置地图索引

                // 如果回环计数超过阈值且漂移大于阈值，触发优化
                if (relc_counts[id] > curr_halt && drift_p > 0.10)
                {
                  isOpt = true; // 标记需要优化
                  for (int &cnt : relc_counts)
                    cnt = 0; // 重置所有回环计数
                }
              }
            }
            else // 如果是跨地图回环
            {
              // 查找地图索引
              for (int i = 0; i < ids.size(); i++)
                if (id == ids[i])
                  step = i;

              // 打印漂移和路程
              printf("drift: %lf %lf\n", drift_p, jours[id]);

              // 如果是新地图（未在优化列表中）
              if (step == -1)
              {
                isGraph = true; // 需要重建图
                isOpt = true; // 需要优化
                relc_counts[id] = 0; // 重置回环计数
                g_update = 1; // 设置重力更新标志
                isPush = true; // 添加约束
                jours[id] = 0; // 重置路程
              }
              else // 已有地图
              {
                // 如果漂移率低于阈值
                if (drift_p / jours[id] < 0.05)
                {
                  jours[id] = 1e-6; // set to 0
                  isPush = true; // 添加约束

                  // 如果回环计数超过阈值且漂移大于阈值，触发优化
                  if (relc_counts[id] > prev_halt && drift_p > 0.25)
                  {
                    isOpt = true; // 标记需要优化
                    // 重置所有回环计数
                    for (int &cnt : relc_counts)
                      cnt = 0;
                  }
                }
              }
            }

            // 如果决定添加约束
            if (isPush)
            {
              match_num++; // 增加匹配计数

              // 添加回环边到边集合
              lp_edges.push(id, cur_id, ord_bl, buf_base - 1, loop_transform.second, loop_transform.first, v6_init);

              // 如果地图索引有效，添加因子图约束
              if (step > -1)
              {
                int id1 = stepsizes[step] + ord_bl; // 第一个位姿ID
                int id2 = stepsizes.back() - 1; // 第二个位姿ID

                // 添加回环约束到因子图
                add_edge(id1, id2, loop_transform.second, loop_transform.first, graph, odom_noise);
                printf("addedge: (%d %d) (%d %d)\n", id, cur_id, ord_bl, buf_base - 1);
              }
            }

            // if(isPush)
            // {
            //   icp_check(*(smp->plptr), *(std_managers[id]->plane_cloud_vec_[search_result.first]), pub_test, pub_init, loop_transform, multimap_scanPoses[id]->at(ord_bl)->x);
            // }
          }
        }
      }

      // 增加所有地图的回环计数
      for (int &it : relc_counts)
        it++;

      // 保存当前描述子
      std_manager->AddSTDescs(stds_vec);

      // 如果需要构建新的位姿图
      if (isGraph)
      {
        build_graph(initial, graph, cur_id, lp_edges, odom_noise, ids, stepsizes, 1);
      }

      // 如果需要执行位姿图优化
      if (isOpt)
      {
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        gtsam::ISAM2 isam(parameters);
        isam.update(graph, initial);

        // gtsam 更新
        for (int i = 0; i < 5; i++)
          isam.update();
        gtsam::Values results = isam.calculateEstimate();
        int resultsize = results.size();

        IMUST x1 = scanPoses->at(buf_base - 1)->x; // 记录优化前的位姿
        int idsize = ids.size(); // 回环检测的session数量，地图数量

        history_kfsize = 0;
        // 根据优化结果更新所有地图中的扫描位姿
        for (int ii = 0; ii < idsize; ii++)
        {
          int tip = ids[ii]; // 当前地图ID
          for (int j = stepsizes[ii]; j < stepsizes[ii + 1]; j++)
          {
            int ord = j - stepsizes[ii]; // 在当前地图中的索引
            // 用优化结果更新扫描位姿
            multimap_scanPoses[tip]->at(ord)->set_state(results.at(j).cast<gtsam::Pose3>());
          }
        }

        // 更新所有关键帧的位姿
        mtx_keyframe.lock();
        for (int ii = 0; ii < idsize; ii++)
        {
          int tip = ids[ii]; // 当前地图ID
          for (Keyframe *kf : *multimap_keyframes[tip])
            kf->x0 = multimap_scanPoses[tip]->at(kf->id)->x;
        }
        mtx_keyframe.unlock();

        // 更新初始值以便下次优化
        initial.clear();
        for (int i = 0; i < resultsize; i++)
          initial.insert(i, results.at(i).cast<gtsam::Pose3>());

        // 计算优化前后的位姿变化
        IMUST x3 = scanPoses->at(buf_base - 1)->x; // 优化后的参考位姿
        dx.p = x3.p - x3.R * x1.R.transpose() * x1.p;
        dx.R = x3.R * x1.R.transpose();
        x_key = x3; // 保存当前关键位姿

        // 将最新的几个关键帧点云添加到体素地图
        PVec pvec_tem;
        int subsize = keyframes->size();
        int init_num = 5; // 处理最新的5个关键帧
        for (int i = subsize - init_num; i < subsize; i++)
        {
          if (i < 0)
            continue;

          // 获取当前关键帧
          Keyframe &sp = *(keyframes->at(i));
          sp.exist = 0; // 标记为已处理
          pvec_tem.reserve(sp.plptr->size()); // 预留空间
          pointVar pv;
          pv.var.setZero();
          for (PointType &ap : sp.plptr->points)
          {
            pv.pnt << ap.x, ap.y, ap.z;
            pv.pnt = sp.x0.R * pv.pnt + sp.x0.p;
            for (int j = 0; j < 3; j++)
              pv.var(j, j) = ap.normal[j];
            pvec_tem.push_back(pv);
          }
          cut_voxel(map_loop, pvec_tem, win_size, 0); // 将点云添加到地图 map_loop
        }

        // 构建关键帧KD树用于后续回环检测
        if (subsize > init_num)
        {
          pl_kdmap->clear();
          for (int i = 0; i < subsize - init_num; i++)
          {
            Keyframe &kf = *(keyframes->at(i));
            kf.exist = 1;
            // 将关键帧位置添加到KD树
            PointType pp;
            pp.x = kf.x0.p[0];
            pp.y = kf.x0.p[1];
            pp.z = kf.x0.p[2];
            pp.intensity = cur_id;
            pp.curvature = i;
            pl_kdmap->push_back(pp);
          }

          kd_keyframes.setInputCloud(pl_kdmap);
          history_kfsize = pl_kdmap->size();
        }
        loop_detect = 1; // 标记回环检测已完成

        // 发布优化后的全局路径和地图用于可视化
        vector<int> ids2 = ids;
        ids2.pop_back();
        ResultOutput::instance().pub_global_path(multimap_scanPoses, pub_prev_path, ids2);
        ResultOutput::instance().pub_globalmap(multimap_keyframes, ids2, pub_pmap);
        ids2.clear();
        ids2.push_back(ids.back());
        ResultOutput::instance().pub_globalmap(multimap_keyframes, ids2, pub_cmap);
      }
    }

    // 清理资源
    for (int i = 0; i < std_managers.size(); i++)
      delete std_managers[i];
    malloc_trim(0);

    // 处理完成后的清理和保存
    if (is_finish)
    {
      if (keyframes->empty())
      {
        sessionNames.pop_back();
        std_managers.pop_back();
        multimap_scanPoses.pop_back();
        multimap_keyframes.pop_back();
        juds.pop_back();
        jours.pop_back();
        relc_counts.pop_back();
      }

      if (multimap_keyframes.empty())
      {
        printf("no data\n");
        return;
      }

      int cur_id = std_managers.size() - 1;
      build_graph(initial, graph, cur_id, lp_edges, odom_noise, ids, stepsizes, 0);

      topDownProcess(initial, graph, ids, stepsizes);
    }

    // 保存地图数据
    if (is_save_map)
    {
      for (int i = 0; i < ids.size(); i++)
        FileReaderWriter::instance().save_pose(*(multimap_scanPoses[ids[i]]), sessionNames[ids[i]], "/alidarState.txt", savepath);

      FileReaderWriter::instance().pgo_edges_io(lp_edges, sessionNames, 1, savepath, bagname);
    }

    // 清理内存
    for (int i = 0; i < multimap_scanPoses.size(); i++)
    {
      for (int j = 0; j < multimap_scanPoses[i]->size(); j++)
        delete multimap_scanPoses[i]->at(j);
    }
    for (int i = 0; i < multimap_keyframes.size(); i++)
    {
      for (int j = 0; j < multimap_keyframes[i]->size(); j++)
        delete multimap_keyframes[i]->at(j);
    }

    malloc_trim(0);
  }

  // The top down process of HBA
  void topDownProcess(gtsam::Values &initial, gtsam::NonlinearFactorGraph &graph, vector<int> &ids, vector<int> &stepsizes)
  {
    cnct_map = ids;
    gba_size = multimap_keyframes.back()->size();
    gba_flag = 1;

    pcl::PointCloud<PointType> pl0;
    pub_pl_func(pl0, pub_pmap);
    pub_pl_func(pl0, pub_cmap);
    pub_pl_func(pl0, pub_curr_path);
    pub_pl_func(pl0, pub_prev_path);
    pub_pl_func(pl0, pub_scan);

    double t0 = ros::Time::now().toSec();
    while (gba_flag) // 等待gba_flag被其他线程置为0
      ;

    for (PGO_Edge &edge : gba_edges1.edges) // 遍历所有第一类地图间约束
    {
      vector<int> step(2); // 存储两个地图的索引位置
      if (edge.is_adapt(ids, step)) // 检查边是否适用于当前地图集合
      {
        int mp[2] = {stepsizes[step[0]], stepsizes[step[1]]}; // 获取两个地图在因子图中的起始索引
        for (int i = 0; i < edge.rots.size(); i++) // 遍历该边的所有约束
        {
          int id1 = mp[0] + edge.ids1[i]; // 计算第一个位姿节点ID
          int id2 = mp[1] + edge.ids2[i]; // 计算第二个位姿节点ID
          // 创建噪声模型，使用约束的协方差信息
          gtsam::noiseModel::Diagonal::shared_ptr v6_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector(edge.covs[i]));
          add_edge(id1, id2, edge.rots[i], edge.tras[i], graph, v6_noise);
        }
      }
    }

    // 与第一类边逻辑相同，处理第二类边
    for (PGO_Edge &edge : gba_edges2.edges)
    {
      vector<int> step(2);
      if (edge.is_adapt(ids, step))
      {
        int mp[2] = {stepsizes[step[0]], stepsizes[step[1]]};
        for (int i = 0; i < edge.rots.size(); i++)
        {
          int id1 = mp[0] + edge.ids1[i];
          int id2 = mp[1] + edge.ids2[i];
          gtsam::noiseModel::Diagonal::shared_ptr v6_noise = gtsam::noiseModel::Diagonal::Variances(gtsam::Vector(edge.covs[i]));
          add_edge(id1, id2, edge.rots[i], edge.tras[i], graph, v6_noise);
        }
      }
    }

    // 使用ISAM2进行位姿图优化
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    gtsam::ISAM2 isam(parameters);
    isam.update(graph, initial);

    for (int i = 0; i < 5; i++)
      isam.update();
    gtsam::Values results = isam.calculateEstimate();
    int resultsize = results.size();

    int idsize = ids.size(); // 获取地图ID数量
    for (int ii = 0; ii < idsize; ii++) // 遍历所有地图
    {
      int tip = ids[ii]; // 当前地图ID
      for (int j = stepsizes[ii]; j < stepsizes[ii + 1]; j++) // 遍历该地图中的所有位姿
      {
        int ord = j - stepsizes[ii]; // 在当前地图中的索引
        multimap_scanPoses[tip]->at(ord)->set_state(results.at(j).cast<gtsam::Pose3>()); // 用优化结果更新位姿
      }
    }

    // 获取第一个地图第一个位姿的四元数（可能用于调试或可视化）
    Eigen::Quaterniond qq(multimap_scanPoses[0]->at(0)->x.R);

    double t1 = ros::Time::now().toSec();
    printf("GBA opt: %lfs\n", t1 - t0);

    // 更新所有关键帧的位姿
    for (int ii = 0; ii < idsize; ii++)
    {
      int tip = ids[ii]; // 当前地图ID
      for (Keyframe *smp : *multimap_keyframes[tip]) // 遍历该地图中的所有关键帧
        smp->x0 = multimap_scanPoses[tip]->at(smp->id)->x; // 用优化结果更新位姿
    }

    ResultOutput::instance().pub_global_path(multimap_scanPoses, pub_prev_path, ids);
    vector<int> ids2 = ids;
    ids2.pop_back();
    ResultOutput::instance().pub_globalmap(multimap_keyframes, ids2, pub_pmap);
    ids2.clear();
    ids2.push_back(ids.back());
    ResultOutput::instance().pub_globalmap(multimap_keyframes, ids2, pub_cmap);
  }

  // The bottom up to add edge in HBA
  // p_xs: 位姿向量，存储所有关键帧的位姿估计(IMUST类型，包含旋转R和平移p)
  // p_smps: 关键帧指针向量，存储所有关键帧
  // gba_edges: 位姿图优化边结构，用于存储添加的约束关系
  // maps: 地图ID向量，指定哪些地图参与优化
  // max_iter: 最大迭代次数
  // thread_num: 线程数量，用于并行计算
  // plptr: 点云指针(可选参数)，用于可视化或存储结果
  void HBA_add_edge(vector<IMUST> &p_xs, vector<Keyframe *> &p_smps, PGO_Edges &gba_edges, vector<int> &maps, int max_iter, int thread_num, pcl::PointCloud<PointType>::Ptr plptr = nullptr)
  {
    bool is_display = false;
    if (plptr == nullptr)
      is_display = true; // 如果没有提供点云指针，设置为显示模式

    double t0 = ros::Time::now().toSec();

    // 筛选属于指定地图的关键帧
    vector<Keyframe *> smps; // 存储筛选后的关键帧
    vector<IMUST> xs; // 存储筛选后的位姿
    int last_mp = -1, isCnct = 0; // last_mp记录上一个地图ID，isCnct标记是否连接

    // 遍历所有关键帧
    for (int i = 0; i < p_smps.size(); i++)
    {
      Keyframe *smp = p_smps[i];
      if (smp->mp != last_mp) // 如果当前关键帧地图ID与上一个不同
      {
        isCnct = 0; // 重置连接标志
        for (int &m : maps) // 检查当前地图是否在指定的地图列表中
          if (smp->mp == m)
          {
            isCnct = 1; // 设置连接标志为1
            break;
          }
        last_mp = smp->mp; // 更新上一个地图ID
      }

      // 如果当前关键帧属于指定地图，则添加到筛选列表中
      if (isCnct)
      {
        smps.push_back(smp);
        xs.push_back(p_xs[i]);
      }
    }

    // 优化参数设置
    int wdsize = smps.size(); // 窗口大小，即筛选后的关键帧数量
    Eigen::MatrixXd hess; // 海森矩阵（信息矩阵）
    // 保存原始全局参数值，以便后续恢复
    vector<double> gba_eigen_value_array_orig = gba_eigen_value_array;
    double gba_min_eigen_value_orig = gba_min_eigen_value;
    double gba_voxel_size_orig = gba_voxel_size;

    int up = 4; // 更新策略参数
    int converge_flag = 0; // 收敛标志
    double converge_thre = 0.05; // 收敛阈值

    for (int iterCnt = 0; iterCnt < max_iter; iterCnt++)
    {
      if (converge_flag == 1 || iterCnt == max_iter - 1)
      {
        // if(plptr == nullptr)
        // {
        //   break;
        // }

        gba_voxel_size = voxel_size;
        gba_eigen_value_array = plane_eigen_value_thre;
        gba_min_eigen_value = min_eigen_value;
      }

      // 构建八叉树体素地图
      unordered_map<VOXEL_LOC, OctreeGBA *> oct_map;
      for (int i = 0; i < wdsize; i++)
        OctreeGBA::cut_voxel(oct_map, xs[i], smps[i]->plptr, i, wdsize); // 将关键帧点云分割到体素地图中

      // 创建激光雷达因子图
      LidarFactor voxhess(wdsize);
      OctreeGBA_multi_recut(oct_map, voxhess, thread_num); // 多线程优化八叉树体素地图

      // 创建激光束调整优化器
      Lidar_BA_Optimizer opt_lsv;
      opt_lsv.thd_num = thread_num; // 设置线程数
      vector<double> resis; // 残差向量
      // 迭代优化
      bool is_converge = opt_lsv.damping_iter(xs, voxhess, &hess, resis, up, is_display);

      // 打印残差变化比例
      if (is_display)
        printf("%lf\n", fabs(resis[0] - resis[1]) / resis[0]);

      // 检查收敛条件
      if ((fabs(resis[0] - resis[1]) / resis[0] < converge_thre && is_converge) || (iterCnt == max_iter - 2 && converge_flag == 0))
      {
        converge_thre = 0.01; // 降低收敛阈值
        if (converge_flag == 0)
        {
          converge_flag = 1;
        }
        else if (converge_flag == 1)
        {
          break;
        }
      }
    }

    // 恢复原始参数值
    gba_eigen_value_array = gba_eigen_value_array_orig;
    gba_min_eigen_value = gba_min_eigen_value_orig;
    gba_voxel_size = gba_voxel_size_orig;

    // 遍历所有关键帧对，添加约束边
    for (int i = 0; i < wdsize - 1; i++)
      for (int j = i + 1; j < wdsize; j++)
      {
        bool isAdd = true; // 是否添加约束边
        Eigen::Matrix<double, 6, 1> v6; // 6自由度约束协方差
        for (int k = 0; k < 6; k++)
        {
          double hc = fabs(hess(6 * i + k, 6 * j + k)); // 提取海森矩阵对应元素
          if (hc < 1e-6) // 小于 1e-6，不添加约束
          {
            isAdd = false;
            break;
          }
          v6[k] = 1.0 / hc;
        }

        // 添加约束边
        if (isAdd)
        {
          Keyframe &s1 = *smps[i];
          Keyframe &s2 = *smps[j];
          Eigen::Vector3d tra = xs[i].R.transpose() * (xs[j].p - xs[i].p);
          Eigen::Matrix3d rot = xs[i].R.transpose() * xs[j].R;
          gba_edges.push(s1.mp, s2.mp, s1.id, s2.id, rot, tra, v6);
        }
      }

    // 如果提供了点云指针，进行点云可视化
    if (plptr != nullptr)
    {
      pcl::PointCloud<PointType> pl; // 存储优化后的点云
      IMUST xc = xs[0]; // 获取第一个关键帧的位姿
      // 计算每个关键帧相对于参考系的变换
      for (int i = 0; i < wdsize; i++)
      {
        Eigen::Vector3d dp = xc.R.transpose() * (xs[i].p - xc.p);
        Eigen::Matrix3d dR = xc.R.transpose() * xs[i].R;

        // 转换每个点到参考坐标系
        for (PointType ap : smps[i]->plptr->points)
        {
          Eigen::Vector3d v3(ap.x, ap.y, ap.z);
          v3 = dR * v3 + dp;
          ap.x = v3[0];
          ap.y = v3[1];
          ap.z = v3[2];
          ap.intensity = smps[i]->mp;
          pl.push_back(ap);
        }
      }

      // 对结果点云进行下采样
      down_sampling_voxel(pl, voxel_size / 8);

      // 更新输出点云
      plptr->clear();
      plptr->reserve(pl.size());
      for (PointType &ap : pl.points)
        plptr->push_back(ap);
    }
    else
    {
      // pcl::PointCloud<PointType> pl, path;
      // pub_pl_func(pl, pub_test);
      // for(int i=0; i<wdsize; i++)
      // {
      //   PointType pt;
      //   pt.x = xs[i].p[0]; pt.y = xs[i].p[1]; pt.z = xs[i].p[2];
      //   path.push_back(pt);
      //   for(int j=1; j<smps[i]->plptr->size(); j+=2)
      //   {
      //     PointType ap = smps[i]->plptr->points[j];
      //     Eigen::Vector3d v3(ap.x, ap.y, ap.z);
      //     v3 = xs[i].R * v3 + xs[i].p;
      //     ap.x = v3[0]; ap.y = v3[1]; ap.z = v3[2];
      //     ap.intensity = smps[i]->mp;
      //     pl.push_back(ap);

      //     if(pl.size() > 1e7)
      //     {
      //       pub_pl_func(pl, pub_test);
      //       pl.clear();
      //       sleep(0.05);
      //     }
      //   }
      // }
      // pub_pl_func(pl, pub_test);
      // return;
    }
  }

  // The main thread of bottom up in global mapping
  void thd_globalmapping(ros::NodeHandle &n)
  {
    n.param<double>("GBA/voxel_size", gba_voxel_size, 1.0);
    n.param<double>("GBA/min_eigen_value", gba_min_eigen_value, 0.01);
    n.param<vector<double>>("GBA/eigen_value_array", gba_eigen_value_array, vector<double>());
    for (double &iter : gba_eigen_value_array)
      iter = 1.0 / iter;
    int total_max_iter = 1;
    n.param<int>("GBA/total_max_iter", total_max_iter, 1);

    vector<Keyframe *> gba_submaps; // 存储全局地图优化的子地图关键帧
    deque<int> localID; // 局部窗口中的关键帧ID队列

    int smp_mp = 0; // 当前处理的地图索引
    int buf_base = 0; // 当前地图中的处理位置
    int wdsize = 10; // 窗口大小，使用10个关键帧进行局部优化
    int mgsize = 5; // 边际化大小，每次移除5个最早的关键帧
    int thread_num = 5; // 线程数量，用于并行计算

    while (n.ok())
    {
      if (multimap_keyframes.empty())
      {
        sleep(0.1);
        continue;
      }

      int smp_flag = 0; // 标记是否需要切换到下一个地图
      if (smp_mp + 1 < multimap_keyframes.size() && !multimap_keyframes.back()->empty())
        smp_flag = 1; // 如果有新地图且不为空，设置标记为1

      vector<Keyframe *> &smps = *multimap_keyframes[smp_mp]; // 获取当前地图的关键帧列表引用

      // 检查是否需要进行全局优化
      int total_ba = 0; // 全局优化标志，0表示不需要全局优化
      if (gba_flag == 1 && smp_mp >= cnct_map.back() && gba_size <= buf_base)
      {
        printf("gba_flag enter: %d\n", gba_flag);
        total_ba = 1; // 如果全局优化标志已设置且当前处理的地图是最新地图且已处理足够多关键帧，则需要进行全局优化
      }
      else if (smps.size() <= buf_base) // 如果当前地图的关键帧已全部处理
      {
        if (smp_flag == 0) // 如果当前地图没有新地图且没有关键帧需要处理，则休眠0.1秒
        {
          sleep(0.1);
          continue;
        }
      }
      else // 如果当前地图还有关键帧需要处理
      {
        smps[buf_base]->mp = smp_mp; // 设置关键帧所属地图索引
        localID.push_back(buf_base); // 将当前关键帧ID添加到局部窗口

        buf_base++; // 增加处理位置
        if (localID.size() < wdsize) // 如果局部窗口中的关键帧数量小于窗口大小，则休眠0.1秒
        {
          sleep(0.1);
          continue;
        }
      }

      // 准备局部优化数据
      vector<IMUST> xs; // 存储局部窗口中关键帧的位姿
      vector<Keyframe *> smp_local; // 存储局部窗口中的关键帧
      mtx_keyframe.lock();
      // 遍历局部窗口中的关键帧ID，添加到位姿列表和关键帧列表中
      for (int i : localID)
      {
        xs.push_back(multimap_keyframes[smp_mp]->at(i)->x0);
        smp_local.push_back(multimap_keyframes[smp_mp]->at(i));
      }
      mtx_keyframe.unlock();

      double tg1 = ros::Time::now().toSec();

      // 创建子地图并添加约束
      Keyframe *gba_smp = new Keyframe(smp_local[0]->x0); // 创建新的子地图关键帧，使用第一个局部关键帧的位姿
      vector<int> mps{smp_mp}; // 创建只包含当前地图索引的数组
      HBA_add_edge(xs, smp_local, gba_edges1, mps, 1, 2, gba_smp->plptr); // 添加局部约束边，同时生成下采样点云
      gba_smp->id = smp_local[0]->id; // 设置子地图关键帧的ID为局部窗口第一个关键帧的ID
      gba_smp->mp = smp_mp; // 设置子地图对应的地图索引
      gba_submaps.push_back(gba_smp); // 将子地图关键帧添加到全局地图优化列表中

      // 全局优化处理
      if (total_ba == 1) // 如果需要进行全局优化
      {
        printf("GBAsize: %d\n", gba_size);
        vector<IMUST> xs; // 存储所有子地图关键帧的位姿
        mtx_keyframe.lock();
        // 遍历所有子地图关键帧
        for (Keyframe *smp : gba_submaps)
        {
          xs.push_back(multimap_scanPoses[smp->mp]->at(smp->id)->x);
        }
        mtx_keyframe.unlock();
        gba_edges2.edges.clear();
        gba_edges2.mates.clear();
        // 对所有子地图进行全局优化，添加地图间约束边
        HBA_add_edge(xs, gba_submaps, gba_edges2, cnct_map, total_max_iter, thread_num);

        if (is_finish) // 如果系统完成标志已设置
        {
          for (int i = 0; i < gba_submaps.size(); i++)
            delete gba_submaps[i]; // 释放子地图内存
        }
        gba_submaps.clear(); // 清空子地图列表

        malloc_trim(0); // 释放未使用的内存
        gba_flag = 0; // 重置全局优化标志
      }
      else if (smp_flag == 1 && multimap_keyframes[smp_mp]->size() <= buf_base) // 如果需要切换到下一个地图且当前地图已处理完
      {
        smp_mp++; // 增加地图索引
        buf_base = 0; // 重置处理位置
        localID.clear(); // 清空局部窗口中的关键帧ID队列
        // printf("switch: %d\n", smp_mp);
      }
      else // 如果当前地图还有关键帧需要处理
      {
        for (int i = 0; i < mgsize; i++)
          localID.pop_front(); // 移除mgsize个最早的关键帧
      }
    }
  }
};

int main(int argc, char **argv)
{
  ros::init(argc, argv, "cmn_voxel");
  ros::NodeHandle n;

  pub_cmap = n.advertise<sensor_msgs::PointCloud2>("/map_cmap", 100);
  pub_pmap = n.advertise<sensor_msgs::PointCloud2>("/map_pmap", 100);
  pub_scan = n.advertise<sensor_msgs::PointCloud2>("/map_scan", 100);
  pub_init = n.advertise<sensor_msgs::PointCloud2>("/map_init", 100);
  pub_test = n.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
  pub_curr_path = n.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
  pub_prev_path = n.advertise<sensor_msgs::PointCloud2>("/map_true", 100);

  VOXEL_SLAM vs(n);
  mp = new int[vs.win_size];
  for (int i = 0; i < vs.win_size; i++)
    mp[i] = i;

  thread thread_loop(&VOXEL_SLAM::thd_loop_closure, &vs, ref(n));
  thread thread_gba(&VOXEL_SLAM::thd_globalmapping, &vs, ref(n));
  vs.thd_odometry_localmapping(n);

  thread_loop.join();
  thread_gba.join();
  ros::spin();
  return 0;
}
