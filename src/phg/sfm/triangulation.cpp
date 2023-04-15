#include "triangulation.h"

#include "defines.h"

#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // составление однородной системы + SVD
    // без подвохов
//    throw std::runtime_error("not implemented yet");
    int eq_num = count * 2;
    Eigen::MatrixXd A(eq_num, 3);
    Eigen::VectorXd b(eq_num);

    for (int i_pair = 0; i_pair < eq_num / 2; ++i_pair) {
        double x = ms[i_pair][0];
        double y = ms[i_pair][1];
        double z = ms[i_pair][2];
        auto psCur = Ps[i_pair];

        for (int i = 0; i < 3; i++) {
            A(i_pair * 2, i) = x * psCur(2, i) - z * psCur(0, i);
            A(i_pair * 2 + 1, i) = y * psCur(2, i) - z * psCur(1, i);
        }

        b(i_pair * 2) = z * psCur(0, 3) - x * psCur(2, 3);
        b(i_pair * 2+ 1) = z * psCur(1, 3) - y * psCur(2, 3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::MatrixXd newD(3, eq_num);
    newD.setZero();
    for (int i = 0; i < 3; i++) {
        newD(i, i) = 1 / svd.singularValues()[i];
    }
    Eigen::VectorXd sol = svd.matrixV() * newD * svd.matrixU().transpose() * b;

    return cv::Vec4d(sol[0], sol[1], sol[2], 1);
}
