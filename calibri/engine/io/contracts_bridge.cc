#include "io/contracts_bridge.h"

namespace ba {

void PoseSE3_To_qt(const PoseSE3& P, Eigen::Quaterniond& q, Eigen::Vector3d& t) {
    // PoseSE3::T is float[16] row-major [R|t; 0 0 0 1]
    Eigen::Matrix4d T = Eigen::Map<const Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(P.T).cast<double>();
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    t = T.block<3, 1>(0, 3);
    q = Eigen::Quaterniond(R).normalized();
}

void PoseSE3_To_qt(const PoseSE3& P, double q_out[4], double t_out[3]) {
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
    PoseSE3_To_qt(P, q, t);
    // Eigen internal order: [x, y, z, w]
    q_out[0] = q.x();
    q_out[1] = q.y();
    q_out[2] = q.z();
    q_out[3] = q.w();
    t_out[0] = t.x();
    t_out[1] = t.y();
    t_out[2] = t.z();
}

PoseSE3 qt_To_PoseSE3(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    PoseSE3 P{};
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    T.block<3, 1>(0, 3) = t;
    Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(P.T) = T.cast<float>();
    return P;
}

PoseSE3 qt_To_PoseSE3(const double q_in[4], const double t_in[3]) {
    // q_in in Eigen internal order: [x, y, z, w]
    // Normalize to guard against floating-point drift (Rule 5)
    Eigen::Quaterniond q = Eigen::Quaterniond(q_in[3], q_in[0], q_in[1], q_in[2]).normalized();
    Eigen::Vector3d t(t_in[0], t_in[1], t_in[2]);
    return qt_To_PoseSE3(q, t);
}

} // namespace ba
