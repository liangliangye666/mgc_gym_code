#ifndef L5A_ACTION_LIMITER_H
#define L5A_ACTION_LIMITER_H

#include <Eigen/Core>

#include <algorithm>
#include <array>
#include <stdexcept>

namespace l5a {

inline Eigen::VectorXd LimitL5aActions(const Eigen::VectorXd& raw_actions,
                                       const Eigen::VectorXd& previous_actions,
                                       double clip_actions,
                                       double joint_delta_limit,
                                       double wheel_delta_limit,
                                       double wheel_abs_limit) {
  constexpr int kNumActions = 8;
  constexpr std::array<int, 2> kWheelIndices{3, 7};
  if (raw_actions.size() != kNumActions || previous_actions.size() != kNumActions) {
    throw std::invalid_argument("L5A action limiter expects exactly 8 actions");
  }

  Eigen::VectorXd limited = raw_actions.cwiseMin(clip_actions).cwiseMax(-clip_actions);
  for (int i = 0; i < kNumActions; ++i) {
    const bool is_wheel = i == kWheelIndices[0] || i == kWheelIndices[1];
    const double delta_limit = is_wheel ? wheel_delta_limit : joint_delta_limit;
    const double delta = std::clamp(limited[i] - previous_actions[i],
                                    -delta_limit, delta_limit);
    limited[i] = previous_actions[i] + delta;
    if (is_wheel) {
      limited[i] = std::clamp(limited[i], -wheel_abs_limit, wheel_abs_limit);
    }
  }
  return limited;
}

}  // namespace l5a

#endif  // L5A_ACTION_LIMITER_H
