
#ifndef LR_H_
#define LR_H_
#include <math.h>
#include "Utils.h"
#include "bignumber.h"
#include "common.h"
#include "string/leco_uint256.h"
struct lr {  // theta0+theta1*x
  double theta0;
  double theta1;
  int delta;


  __device__ void caltheta(std::vector<double>& x, std::vector<double>& y, int m) {
    double sumx = 0;
    double sumy = 0;
    double sumxy = 0;
    double sumxx = 0;
    for (int i = 0; i < m; i++) {
      sumx = sumx + x[i];
      sumy = sumy + y[i];
      sumxx = sumxx + x[i] * x[i];
      sumxy = sumxy + x[i] * y[i];
    }

    double ccc = sumxy * m - sumx * sumy;
    double xxx = sumxx * m - sumx * sumx;

    theta1 = ccc / xxx;
    theta0 = (sumy - theta1 * sumx) / (double)m;
  }

  __device__ void caltheta_LOO(double x[], double y[], int m) {
    double sumx = Utils::array_sum(x, m);
    double sumy = Utils::array_sum(y, m);
    double xy = Utils::array_sum(Utils::array_multiplication(x, y, m), m);
    double xx = Utils::array_sum(Utils::array_multiplication(x, x, m), m);
    for (int i = 0; i < m; i++) {
      double tmpavx = (sumx - x[i]) * ((double)1 / (m - 1));
      double tmpavy = (sumy - y[i]) * ((double)1 / (m - 1));
      double tmpxy = xy - x[i] * y[i];
      double tmpxx = xx - x[i] * x[i];
      theta1 = (tmpxy - (double(m - 1) * tmpavx * tmpavy)) /
               (tmpxx - (double(m - 1) * tmpavx * tmpavx));
      theta0 = tmpavy - theta1 * tmpavx;
      std::cout << "Theta1: " << theta1 << "Theta0: " << theta0 << std::endl;
    }
  }
  __device__ bool agree(double x, double y) {
    double tmp = abs((x * theta1 + theta0) - y);
    if (tmp <= (double)delta * delta) {
      return true;
    } else {
      return false;
    }
  }
};

template <typename T>
struct lr_int_T{//theta0+theta1*x
    double theta0;
    double theta1;

    
__device__ void caltheta(const T *y, int m){

    double sumx = 0;
    double sumy = 0;
    double sumxy = 0;
    double sumxx = 0;
    for(int i=0;i<m;i++){
        sumx = sumx + (double)i;
        sumy = sumy + (double)y[i];
        sumxx = sumxx+(double)i*i;
        sumxy = sumxy+(double)i*y[i];
    }
    
    double ccc= sumxy * m - sumx * sumy;
    double xxx = sumxx * m - sumx * sumx;

    theta1 = ccc/xxx;
    theta0 = (sumy - theta1 * sumx)/(double)m;
    
}

};

struct lr_int{//theta0+theta1*x
    double theta0;
    double theta1;
  
   __device__ void caltheta(uint32_t *y, int m){

    double sumx = 0;
    double sumy = 0;
    double sumxy = 0;
    double sumxx = 0;
    for(int i=0;i<m;i++){
        sumx = sumx + (double)i;
        sumy = sumy + (double)y[i];
        sumxx = sumxx+(double)i*i;
        sumxy = sumxy+(double)i*y[i];
    }
    
    double ccc= sumxy * m - sumx * sumy;
    double xxx = sumxx * m - sumx * sumx;

    theta1 = ccc/xxx;
    theta0 = (sumy - theta1 * sumx)/(double)m;
    
}

};
#endif
