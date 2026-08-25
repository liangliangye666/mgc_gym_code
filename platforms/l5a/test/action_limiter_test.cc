#include "../control/action_limiter.h"

#include <cassert>
#include <cmath>

int main() {
  Eigen::VectorXd previous(8);
  previous << 0.0, 0.1, -0.1, 19.5, 0.0, -0.1, 0.1, -19.5;
  Eigen::VectorXd raw(8);
  raw << 1.0, -1.0, 0.0, 30.0, -1.0, 1.0, 0.0, -30.0;

  const Eigen::VectorXd limited =
      l5a::LimitL5aActions(raw, previous, 100.0, 0.2, 1.0, 20.0);
  Eigen::VectorXd expected(8);
  expected << 0.2, -0.1, 0.0, 20.0, -0.2, 0.1, 0.0, -20.0;

  assert((limited - expected).cwiseAbs().maxCoeff() < 1e-12);
  return 0;
}
