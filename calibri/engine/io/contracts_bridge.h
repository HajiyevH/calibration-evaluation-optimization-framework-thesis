#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "engine/contracts/types.h"

namespace ba {

// Convert PoseSE3 (float[16] row-major) to Eigen quaternion + translation (double)
void PoseSE3_To_qt(const PoseSE3& P, Eigen::Quaterniond& q, Eigen::Vector3d& t);

// Convert PoseSE3 to raw double arrays: q_out[4] in Eigen internal order [x,y,z,w], t_out[3]
void PoseSE3_To_qt(const PoseSE3& P, double q_out[4], double t_out[3]);

// Convert Eigen quaternion + translation to PoseSE3
PoseSE3 qt_To_PoseSE3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);

// Convert raw double arrays to PoseSE3: q_in[4] in Eigen internal order [x,y,z,w], t_in[3]
PoseSE3 qt_To_PoseSE3(const double q_in[4], const double t_in[3]);

} // namespace ba
