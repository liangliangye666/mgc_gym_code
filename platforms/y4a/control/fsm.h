#ifndef FSM_L4A_H
#define FSM_L4A_H

#ifndef GLOG_USE_GLOG_EXPORT
#  define GLOG_USE_GLOG_EXPORT
#endif

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

#include "fsm_id.h"
#include "rl.h"
#include "robot_model.h"
#include "ros_publisher.h"

#include <glog/logging.h>
namespace y4a {

class FSM {
 public:
  FSM(RobotModel& robot_model);
  FSM() = default;
  ~FSM();
  void Initialize(RobotModel& robot_model);
  void Run(RobotModel& robot_model);
  bool CheckSafety(RobotModel& robot_model);
  Eigen::VectorXd tau() { return tau_; }

#if SIM_ENABLE
  std::shared_ptr<ROS::Publisher> ros_publisher_;
#endif

 private:
  int fsm_id_;
  int last_fsm_id_;
  bool state_changed_;
  bool EDamp_signal_;
  bool safe_mode_;
  YAML::Node config_;

  std::shared_ptr<RL> rl_;

  Eigen::VectorXd tau_;
};

}  // namespace y4a
#endif