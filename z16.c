/*
   Low-Rank Decomposition-Based Restoration of
   Compressed Images via Adaptive Noise Estimation
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "com.h"
#include "dct.h"
#include "qcs.h"
#include "qtz.h"

// extern void dgesvd(char *, char *, const int *, const int *, double *, const int *, double *, double *, const int *, double *, const int *, double *, int *, int *);
// マクロ
#define imgAcc2(i, j, maxi, maxj) ((i) * (maxj) + (j))
#define imgAcc3(i, j, k, maxi, maxj, maxk) ((i) * (maxj) * (maxk) + (j) * (maxk) + (k)) 
#define imgAcc4(i, j, k, l, maxi, maxj, maxk, maxl) ((i) * (maxj) * (maxk) * (maxl) + (j) * (maxk) * (maxl) + (k) * (maxl) + (l))
#define dmalloc(a) ((double *)malloc(sizeof(double) * (a)))
#define imalloc(a) ((int * )malloc(sizeof(int) * (a)))


static double Sigma_nGi[64] =
{
    8.087, 6.312, 4.876, 3.601, 2.744, 2.213, 1.901, 1.737,
    2.068, 1.614, 1.247, 0.921, 0.702, 0.566, 0.486, 0.444, 
    0.712, 0.555, 0.429, 0.317, 0.241, 0.195, 0.167, 0.513,
    0.341, 0.266, 0.206, 0.152, 0.116, 0.093, 0.080, 0.073,
    0.215, 0.168, 0.130, 0.096, 0.073, 0.059, 0.051, 0.046,
    0.156, 0.122, 0.094, 0.069, 0.053, 0.043, 0.037, 0.033,
    0.127, 0.099, 0.076, 0.056, 0.043, 0.035, 0.030, 0.027,
    0.112, 0.088, 0.068, 0.050, 0.038, 0.031, 0.026, 0.024
};

void z16(int w, int h, double *in, double *out)
{
    int i, j, i0, j0, vi, vj, ni, nj, l, n, t;

    //int Mq[64];
    //double sdot = 0.0;
    double sigmae = 0.0;
    double sigmas = 0.0;

    double *y = dmalloc(w * h);  // JPEG DCT係数
    double *x = dmalloc(w * h);  // JPEG 
    double *newx = dmalloc(w * h);
    double *cpyx = dmalloc(w * h);
    double *win = dmalloc(w * h);
    double *Ax = dmalloc(w * h);
    double *lhat = dmalloc(w * h);
    double *uhat = dmalloc(w * h);
    double *tmpx = (double *)malloc(sizeof(double) * w * h);
  

    // y <- JPEGDCT係数
    ibdct(w, h, in, y);
  //  cpyQg(Mq); 

    // Set parameters
    const int Ps = 36; // patch size
    const int K = 60;
    const int Ws = 30; // search range
    const int T = 5;
    const int L = 7;
    const int S = 3;   // over lapping length
    const double alpha = 0.01;
    const double gamma = 0.2;

    if(Ps > K || K > Ws * Ws || S < 0){
        fprintf(stderr, "error : invalid parameter value\n");
        exit(1);
    }

    // Set x^0
    memcpy(x, y, sizeof(double) * w * h);

    // Calculate(6)
    //sdot = (double)(Mq[0] + Mq[1] + Mq[2] + Mq[8] + Mq[9] + Mq[10] + Mq[16] + Mq[17] + Mq[18]) / 9.0;
    //simgae = 1.195 * pow(sdot, 0.6394) + 0.9693;

    getL(w, h, in, lhat, 0.25);
    getU(w, h, in, uhat, 0.25);

    const int rtPs = (int)sqrt((double)Ps);
    const int stp = rtPs - S;
    for(i0 = 0 , ni = 0 ; i0 < w ; i0 += stp, ni++);
    for(j0 = 0 , nj = 0 ; j0 < h ; j0 += stp, nj++);
    
    int *Ng   = imalloc(K * ni * nj);     // Ng（ブロックマッチング後の上位K個のベクトル）
    int *chk  = imalloc(Ws * Ws);
    double *Gi        = dmalloc(ni * nj * K * Ps);                    // 上位のパッチ
    double *tmpGi       = dmalloc(K * Ps);                              // one patch 
    double *err       = dmalloc(Ws * Ws);
    double *ref       = dmalloc(Ps);
    double *tgt       = dmalloc(Ps);
    double *YGi       = dmalloc(Ps * K);     // 元XGi equ.(4)
    double *y_bar     = dmalloc(ni * nj * Ps); // aberage YGi
    double *XGi       = dmalloc(Ps * K);     // 元ZGk
    double *cpyXGi    = dmalloc(Ps * K);
    double *tmpXGi    = dmalloc(Ps * K);
    double *ph        = dmalloc(ni * nj);    // equ.(27)
    double *pv        = dmalloc(ni * nj);    // equ.(28)
    double *mu        = dmalloc(ni * nj);    // equ.(29)
    double *sigma_xGi = dmalloc(ni * nj);    // equ.(30)
    double *sigma_nGi = dmalloc(ni * nj);    // equ.(33)
    double *sigma2_nGi= dmalloc(ni * nj);    // σ_n^2
    double *omega_Gi  = dmalloc(ni * nj * Ps);  // equ.(34)
    double *S2        = dmalloc(ni * nj);    //    equ.(34)
    double *lambda    = dmalloc(Ps);
    double *lambda_kGi = dmalloc(Ps);
    double *lambda_dash= dmalloc(Ps);
    double *tau_kGi    = dmalloc(Ps);
    double *tmpYGi      = dmalloc(Ps);

    

    // SVD preliminary
    int info, lwork;
    double wkopt, *work;
    double *s = dmalloc(Ps);
    double *u = dmalloc(Ps * Ps);
    double *vt = dmalloc(K * K);
    double *us = dmalloc(Ps * K);   // u * s
    double *tmps = dmalloc(Ps * K); // temporary s
     
    lwork = -1;
    dgesvd_("All", "All", &Ps, &K, YGi, &Ps, s, u, &Ps, vt, &K, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    work = dmalloc(lwork * sizeof(double));

    for(t = 0 , sigmas = sigmae; t < T ; t++){  printf("%d\n", t);
        memcpy(cpyx, x, sizeof(double) * w * h);
        printf("Algorithm1\n");
        // Patch clustering
        for(j0 = 0 ; j0 < h ; j0 += stp){
            //printf("%d\n",j0);
            for(i0 = 0 ; i0 < w ; i0 += stp){
                //printf("i0\n");
               // printf("%d\n",i0);
                memset(chk, 0, sizeof(int ) * Ws * Ws);
                memset(err, 0, sizeof(double) * Ws * Ws);

                // Copy target
              //  printf("Copy target\n");
                for(j = 0 ; j < rtPs ; j++){
                    for(i = 0 ; i < rtPs ; i++){
                        tgt[imgAcc2(j, i, rtPs, rtPs)] = x[(j0+j)%h * w + (i0+i)%w];
                    }
                }
               // printf("Block Matching\n");
                // Block matching
                for(vj = -Ws/2 ; vj < Ws/2 ; vj++){
                    for(vi = -Ws/2 ; vi < Ws/2 ; vi++){
                        for(j = 0 ; j < rtPs ; j++){
                            for(i = 0 ; i < rtPs ; i++){
                                ref[j*rtPs + i] = x[(j0+j+vj+h)%h * w + (i0+i+vi+w)%w];
                            }
                        }
                        for(j = 0 ; j < rtPs ; j++){
                            for(i = 0 ; i < rtPs; i++){
                                // Note :err[(Ws/2)*Ws+(Ws/2)], ie, vj = vi = 0 is always 0
                                err[(vj + Ws / 2) * Ws + vi + Ws / 2] += (tgt[imgAcc2(j, i, rtPs, rtPs)] - ref[imgAcc2(j, i, rtPs, rtPs)]) * (tgt[imgAcc2(j, i, rtPs, rtPs)] - ref[imgAcc2(j, i, rtPs, rtPs)]);
                                //err[(vj + Ws / 2) * Ws + vi + Ws / 2] += (tgt[j*rtPs] - ref[j*rtPs+ i]) * (tgt[j*rtPs + i] - ref[j * rtPs + i]);
                            }
                        }
                    }
                }
                //printf("Construct Gi\n");
                // Construct Gi
                for(n = 0 ; n < K ; n++){
                    int bstvi = Ws/2;
                    int bstvj = Ws/2;
                    double minE = 1E+8;
                    for(vj = 0 ; vj < Ws ; vj++){
                        for(vi = 0 ; vi < Ws ; vi++){
                            if(!chk[imgAcc2(vj, vi, Ws, Ws)] && minE > err[imgAcc2(vj, vi, Ws, Ws)]){
                                bstvi = vi;
                                bstvj = vj;
                                minE = err[imgAcc2(vj, vi, Ws, Ws)];
                            }
                        }
                    }
                    Ng[imgAcc3(j0/stp, i0/stp, n, nj, ni, K)] = bstvj * Ws + bstvi;
                    chk[imgAcc2(bstvj, bstvi, Ws, Ws)] = 1;
                }
                //printf("Gi");
                // Gi (平均のパッチを作る)
                for(n = 0 ; n < K ; n++){
                    vi = Ng[K * (j0 / stp * ni + i0 / stp) + n] % Ws;
                        vj = (Ng[K * (j0 / stp * ni + i0 / stp) + n] - vi) / Ws;
                        vi = vi - Ws / 2;
                        vj = vj - Ws / 2;
                        for (j = 0; j < rtPs; j++) {
                            for (i = 0; i < rtPs; i++)
                            {
                             //   XGk[n * Bs + (j * rtBs + i)] = cpyx[(j0 + j + vj + h) % h * w + (i0 + i + vi + w) % w];
                                Gi[imgAcc4(j0/stp, i0/stp, n, j*rtPs+i, nj, ni, K, Ps)] = cpyx[(j0 + j + vj + h) % h * w + (i0 + i + vi + w) % w];
                                tmpGi[imgAcc2(n, j*rtPs+i, K, Ps)] = cpyx[(j0 + j + vj + h) % h * w + (i0 + i + vi + w) % w];
                            }
                        }
                }
                
                for(n = 0 ; n < K ; n++){
                    memset(tmpGi, 0, sizeof(double) * Ps);
                    for(j = 0 ; j < Ps ; j++){
                        tmpYGi[j] += tmpGi[imgAcc2(n, j, K, Ps)];
                    }
                // y_bar
                    for(j = 0 ; j  < Ps ; j++){
                        y_bar[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)] = tmpYGi[j]/ (double)K;
                    }
                }
                 //printf("mu");
                // μ ph, pv 式(27), (28)
                double tmpmu = 0.;
                for(j = 0 ; j < Ps ; j++){
                    tmpmu += y_bar[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)];
                }
                mu[imgAcc2(j0/stp, i0/stp, nj, ni)] = tmpmu / Ps;

                double tmp1, tmp2;
                double phNumerator = 0; 
                double pvNumerator = 0;
                double Denominator = 0;
               
                for(j = 0 ; j < rtPs ; j++){
                    for(i = 0 ; i < rtPs ; i++){
                        tmp1 = y_bar[imgAcc4(j0/stp, i0/stp, j, i, nj, ni, rtPs, rtPs)] - mu[imgAcc2(j0/stp, i0/stp, nj, ni)];
                        phNumerator += (y_bar[imgAcc4(j0/stp, i0/stp, j, (i+1)%rtPs, nj, ni, rtPs, rtPs)] 
                                        - mu[imgAcc2(j0/stp, i0/stp, nj, ni)])*(tmp1);
                        pvNumerator += (y_bar[imgAcc4(j0/stp, i0/stp, (j+1)%rtPs, i, nj, ni, rtPs, rtPs)] 
                                        - mu[imgAcc2(j0/stp, i0/stp, nj, ni)])*(tmp1);
                        Denominator += tmp1 * tmp1;
                    }
                }
                ph[imgAcc2(j0/stp, i0/stp, nj, ni)] = phNumerator / Denominator;
                pv[imgAcc2(j0/stp, i0/stp, nj, ni)] = phNumerator / Denominator;
                tmp1 = tmp2 = 0;
                // 式(30) yGi(i, j) - mu 
                for(j = 0 ; j < Ps ; j++){
                    tmp1 = y_bar[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)] - mu[imgAcc2(j0/stp, i0/stp, nj, ni)];
                    tmp2 += tmp1 * tmp1;
                }

                sigma_xGi[imgAcc2(j0/stp, i0/stp, nj, ni)] = alpha * sqrt(tmp2);
                // printf("31 32\n");
                // equ(31), (32)保留
                // equ(33), (34)
                tmp1 = 0; 
                for(j = 0 ; j < Ps ; j++){

                    //tmp1 += Sigma_nGi[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)];
                    tmp1 += Sigma_nGi[j];
                }
                //printf("S\n");
                S2[imgAcc2(j0/stp, i0/stp, nj, ni)] = tmp1;
                for(j = 0 ; j < Ps ; j++){
                    omega_Gi[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)] = Sigma_nGi[j] / S;
                }
                //printf("33\n");
                // (33)
                tmp1 = 0;
                for(j = 0 ; j < Ps ; j++){
                    tmp1 += omega_Gi[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)] * Sigma_nGi[j];
                }
                //printf("sigma\n");
                sigma_nGi[imgAcc2(j0/stp, i0/stp, nj, ni)] = tmp1;
            } // i0
        } //j0
    
        printf("Algoririthm2\n");
        // Algorithm 2
        memcpy(cpyx, x, sizeof(double) * w * h);

        // 特異値分解
        for(l = 0 ; l < L ; l++){
            printf("%d\n", l);
            memset(win, 0, sizeof(double) * w * h);
            memset(newx, 0, sizeof(double) * w * h);
            memcpy(tmpx, x, sizeof(double) * w * h);

            for(j0 = 0 ; j0 < h ; j0 += stp){
                for(i0 = 0 ; i0 < w ; i0 += stp){

                    // Construct XGi
                    for(n = 0 ; n < K ; n++){
                        vi = Ng[imgAcc3(j0/stp, i0/stp, n, nj, ni, K)] % Ws;
                        vj = (Ng[imgAcc3(j0/stp, i0/stp, n, nj, ni, K)] - vi)/Ws;
                        vi = vi - Ws/2;
                        vj = vj - Ws/2;
                        for(j = 0 ; j < rtPs ; j++){
                            for(i = 0 ; i < rtPs ; i++){
                                YGi[n * Ps + (j * rtPs + i)] = cpyx[(j0 + j + vj + h) % h * w + (i0 + i + vi + w) % w];
                                XGi[n * Ps + (j * rtPs + i)] = tmpx[(j0 + j + vj + h) % h * w + (i0 + i + vi + w) % w];
                            }
                        }
                    }
                    
                    double _sigmas = 0.0;
                    for(i = 0 ; i < Ps * K ; i++){
                        _sigmas += (YGi[i] - XGi[i]) * (YGi[i] - XGi[i]);
                    }
                    memcpy(tmpXGi, XGi, sizeof(double) * Ps * K);
                    // s -> sigma ここでいうλ
                    dgesvd_("All", "All", &Ps, &K, tmpXGi, &Ps, s, u, &Ps, vt, &K, work, &lwork, &info);
                    
                    // soft threshold => 
                    sigma2_nGi[imgAcc2(j0/stp, i0/stp, nj, ni)] = sigma_nGi[imgAcc2(j0/stp, i0/stp, nj, ni)] * sigma_nGi[imgAcc2(j0/stp, i0/stp, nj, ni)];

                    // λ 
                    for(i = 0 ; i < Ps ; i++){
                        lambda_kGi[i] = s[i];
                    }
                    // equ(14)
                    for(i = 0 ;  i < Ps ; i++){
                        lambda_dash[i] = sqrt(s[i] - sigma2_nGi[imgAcc2(j0/stp, i0/stp, nj, ni)]);
                    }
                    // equ(13)
                    for(i = 0 ; i < Ps ; i++){
                        tau_kGi[i] = gamma * sigma2_nGi[imgAcc2(j0/stp, i0/stp, nj, ni)] / sqrt(lambda_dash[i]);
                    }
                    // equ(12)
                    for(i = 0 ; i < Ps ; i++){
                        if(abs(lambda[i]) > tau_kGi[i]){
                            tmps[i] = lambda[i] - tau_kGi[i] * 1;
                        }else if(abs(lambda[i]) <= tau_kGi[i]){
                            tmps[i] = 0;
                        }
                    }

                    // パッチの再構成 U * D * Vt equ(9)
                    memset(us, 0, sizeof(double) * Ps * K);
                    memset(XGi, 0, sizeof(double) * Ps * K);
                    
                    for(j = 0 ; j < Ps ; j++){
                        for(i = 0 ; i < Ps ; i++){
                            us[K * j + i] = u[Ps * i + j] * tmps[i];
                        }
                    }

                    for(j = 0 ; j < Ps ; j++){
                        for(i = 0 ; i < K ; i++){
                            for(n = 0 ; n < K ; n++){
                                XGi[Ps * i + j] += us[Ps * j + n] * vt[K * i + n];
                            }
                        }
                    }
                    // (15)
                    for(j = 0 ; j < rtPs ; j++){
                        for(i = 0 ; i < rtPs ; i++){
                            newx[(j0 + j) % h * w + (i0 + i) % w] += XGi[rtPs * j + i];
                            win[(j0 + j)%h * w + (i0 + i) % w]++;
                        }
                    }
                } // j0
            } // i0
            // M = min(Ps, K);
            // omega_B = max(1 - r / M, 1/M) / Z;
            for(i = 0 ; i < w * h ; i++){
                x[i] = newx[i] / win[i];
            }
        } // l
        bdct(w, h, x, Ax);
        for(i = 0 ; i < w ; i++){
            Ax[i] = MAX(MIN(Ax[i], uhat[i]), lhat[i]);
        }
        ibdct(w, h, Ax, x);
    }
    memcpy(out, x, sizeof(double) * w * h);
    free(y);
    free(x);
    free(newx);
    free(cpyx);
    free(win);
    free(lhat);
    free(uhat);
    free(tmpx);
    free(tmpGi);
    free(Gi);
    free(chk);
    free(err);
    free(ref);
    free(tgt);
    free(YGi);
    free(XGi);
    free(cpyXGi);
    free(ph);
    free(pv);
    free(mu);
    free(sigma_xGi);
    free(sigma_nGi);
    free(sigma2_nGi);
    free(omega_Gi);
    free(S2);
    free(us);
    free(tmps);
    free(s);
    free(u);
    free(vt);
    free(work);
    free(lambda);
    free(lambda_dash);
    free(lambda_kGi);
    free(tmpYGi);
}