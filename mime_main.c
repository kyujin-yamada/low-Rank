#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "com.h"
#include "dct.h"
#include "pgm.h"
#include "qtz.h"

// マクロ
#define imgAcc2(i, j, maxi, maxj) ((i) * (maxj) + (i))
#define imgAcc3(i, j, k, maxi, maxj, maxk) ((i) * (maxj) * (maxk) + (j) * (maxk) + (k)) 
#define imgAcc4(i, j, k, l, maxi, maxj, maxk, maxl) ((i) * (maxj) * (maxk) * (maxl) + (j) * (maxk) * (maxl) + (k) * (maxl) + (l))
#define dmalloc(a) ((double *)malloc(sizeof(double) * (a)))
#define imalloc(a) ((int * )malloc(sizeof(int) * (a)))

static double Sigma_n[64] =
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
// 特異値分解

double psnr(PGM src, PGM dst);
void lowRank(int, int , double *, double *);

int main(int argc, char **argv){
    int i;

    PGM src, dst;

    if(argc != 3){
        fprintf(stderr, "usage : %s inputfile quality\n ", argv[0]);
        exit(EXIT_FAILURE);
    }

    if(!pgmread(&src, argv[1])){
        fprintf(stderr, "error: can't open %s\n", argv[1]);
        exit(EXIT_FAILURE);
    }

    sclQt(atoi(argv[2]));

    const int w = src.width;
    const int h = src.height;
    int *piQtzIdx = (int *)malloc(sizeof(int) * w * h);
    double *pdSrcImg = dmalloc(w * h);
    double *pdDstImg = dmalloc(w * h);
    double *pdSrcCff = dmalloc(w * h);
    double *pdDstCff = dmalloc(w * h);

    // cIMAGE
    pgmcreate(&dst, w, h);
    
    for(i = 0 ; i < w * h ; i++){
        pdSrcImg[i] = src.data[i];
    }

    // DCT -> Q -> IDCT -> JPEG : 左 in 右 out
    bdct(w, h, pdSrcImg, pdSrcCff);
    bqtz(w, h, pdSrcCff, piQtzIdx);
    ibqtz(w, h, piQtzIdx, pdDstCff);
    ibdct(w, h, pdDstCff, pdDstImg);

    // qtz
    for(i = 0 ; i < w * h ; i++){
        dst.data[i] = (int)(MAX(MIN(((int)(pdDstImg[i] + 0.5)), 255), 0));
    }
    // 問題を解く
    lowRank(w, h, pdDstCff, pdDstImg);

    pgmwrite(dst, "dst_our.pgm");

    free(piQtzIdx);
    free(pdSrcImg);
    free(pdDstImg);
    free(pdSrcCff);
    free(pdDstCff);
    pgmdestroy(&src);
    pgmdestroy(&dst);

    return 0;
}

double psnr(PGM src, PGM dst){
    int i;
    double e;

    if(src.width != dst.width || src.height != dst.height){
        fprintf(stderr, "error : src and dst must be of same size\n");
        exit(1);
    }

    const int w = src.width;
    const int h = src.height;

    for(i = 0 , e = 0.0 ; i < w * h ; i++){
        e += (src.data[i] - dst.data[i]) * (src.data[i] - dst.data[i]);
    }
    return 10 * log10(255 * 255 / (e / (double)(w * h)));
}

void lowRank(int w, int h, double *in, double *out)
{
    int i, j, i0, j0, vi, vj, ni, nj, l, n, t;

    int Mq[64];
    double sdot = 0.0;
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

    // y <- JPEGDCT係数
    ibdct(w, h, in, y);
    cpyQg(Mq); 

    // Set parameters
    const int Ps = 64; // patch size
    const int K = 40;
    const int Ws = 30; // search range
    const int T = 5;
    const int L = 7;
    const int S = 3;   // over lapping length

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
    
    int *Gi = imalloc(K * ni * nj); 
    int *chk = imalloc(Ws * Ws);
    double *err = dmalloc(Ws * Ws);
    double *ref = dmalloc(Ps);
    double *tgt = dmalloc(Ps);
    double *YGi = dmalloc(Ps * K);   // 元Xgi
    double *XGi = dmalloc(Ps * K);   // 元ZGk
    double *cpyXGi = dmalloc(Ps * K);
    double *tmpXGi = dmalloc(Ps * K);
    double *tmps = dmalloc(Ps * K);
    

    // SVD preliminary
    int info, lwork;
    double wkopt, *work;
    double *s = dmalloc(Ps);
    double *u = dmalloc(Ps * Ps);
    double *vt = dmalloc(K * K);

    lwork = -1;
    dgesvd("All", "All", &Ps, &K, YGi, &Ps, s, u, &Ps, vt, &K, &wkopt, &lwork, &info);
    lwork = (int)wkopt;
    work = dmalloc(lwork * sizeof(double));

    for(t = 0 , sigmas = sigmae; t < T ; t++){
        // Patch clustering
        for(j0 = 0 ; j0 < h ; j0 += stp){
            for(i0 = 0 ; i0 < w ; i0 += stp){
                memset(chk, 0, sizeof(int ) * Ws * Ws);
                memset(err, 0, sizeof(double) * Ws * Ws);

                // Copy target
                for(j = 0 ; j < rtPs ; j++){
                    for(i = 0 ; i < rtPs ; i++){
                        tgt[imgAcc2(j, i, rtPs, rtPs)] = x[imgAcc2((j0+j)%h, (i0+i)%w, w, w)];
                    }
                }

                // Block matching
                for(vj = -Ws/2 ; vj < Ws/2 ; vj++){
                    for(vi = -Ws/2 ; vi < Ws/2 ; vi++){
                        for(j = 0 ; j < rtPs ; j++){
                            for(i = 0 ; i < rtPs ; i++){
                                ref[imgAcc2(j, i, rtPs, rtPs)] = x[imgAcc2((j0+j+vj+h)%h, (i0+i+vi+w)%w, w, w)];
                            }
                        }

                        for(j = 0 ; j < rtPs ; j++){
                            for(i = 0 ; i < rtPs; ){
                                // Note :err[(Ws/2)*Ws+(Ws/2)], ie, vj = vi = 0 is always 0
                                err[imgAcc2((vj+Ws/2), vi+Ws/2, Ws, Ws)] += (tgt[imgAcc2(j, i, rtPs, rtPs)]) * (tgt[imgAcc2(j, i, rtPs, rtPs)]);
                            }
                        }
                    }
                }
            
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
                    Gi[imgAcc3(j0/stp, i0/stp, n, nj, ni, K)] = bstvj * Ws + bstvi;
                    chk[imgAcc2(bstvj, bstvi, Ws, Ws)] = 1;
                }

                // YGi (平均のパッチを作る)
                double tmpYGi;
                for(j = 0 ; j < Ps ; j++){
                    tmpYGi = 0.;
                    for(n = 0 ; n < K ; n++){
                        tmpYGi += Gi[imgAcc4(j0/stp, i0/stp, j, n, nj, ni, Ps, n)];
                    }   
                // y_bar
                YGi[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)] = tmpYGi/K;
                }

                // μ ph, pv 式(27), (28)
                double tmpmu = 0.;
                for(j = 0 ; j < Ps ; j++){
                    tmpmu += YGi[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)];
                }
                mu[imgAcc2(j0/stp, i0/stp, nj, ni)] = tmpmu / Ps;

                double tmpPh=0, tmpPv=0, tmp1, tmp2;
                double phNumerator = 0; 
                double pvNumerator = 0;
                double Denominator = 0;
                double ph = 0, pv = 0;
                for(j = 0 ; j < rtPs ; j++){
                    for(i = 0 ; i < rtPs ; i++){
                        tmp1 = YGi[imgAcc4(j0/stp, i0/stp, j, i, nj, ni, rtPs, rtPs)] - mu[imgAcc2(j0/stp, i0/stp, nj, ni)];
                        phNumerator += (YGi[imgAcc4(j0/stp, i0/stp, j, (i+1)%rtPs, nj, ni, rtPs, rtPs)] 
                                        - mu[imgAcc2(j0/stp, i0/stp, nj, ni)])*(tmp1);
                        pvNumerator += (YGi[imgAcc4(j0/stp, i0/stp, (j+1)%rtPs, i, nj, ni, rtPs, rtPs)] 
                                        - mu[imgAcc2(j0/stp, i0/stp, nj, ni)])*(tmp);
                        Denominator += tmp1 * tmp1;
                    }
                }
                ph[imgAcc2(j0/stp, i0/stp, nj, ni)] = phNumerator / Denominator;
                pv[imgAcc2(j0/stp, i0/stp, nj, ni)] = phNumerator / Denominator;
                tmp1 = tmp2 = 0;
                // 式(30) yGi(i, j) - mu 
                for(j = 0 ; j < Ps ; j++){
                    tmp1 = YGi[imgAcc4(j0/stp, i0/stp, j, i, nj, ni, rtPs, rtPs)] - mu[imgAcc2(j0/stp, i0/stp, nj, ni)];
                    tmp2 += tmp1 * tmp1;
                }

                sigmaxGi[imgAcc2(j0/stp, i0/stp, nj, ni)] = alpha * sqrt(tmp2);

                // equ(31), (32)保留
                // equ(33), (34)
                tmp = 0; 
                for(j = 0 ; j < Ps ; j++){
                    tmp += Sigma_nGi[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)];
                }
                S[imgAcc2(j0/stp, i0/stp, nj, ni)] = tmp;
                for(j = 0 ; j < Ps : j++){
                    wGi[imgAcc(j0/stp, i0/stp, j, nj, ni, Ps)] = Sigma_nGi[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)] / S;
                }
                // (33)
                tmp = 0;
                for(j = 0 ; j < Ps ; j++){
                    tmp += wGi[imgAcc(j0/stp, i0/stp, j, nj, ni, Ps)] * Sigma_nGi[imgAcc3(j0/stp, i0/stp, j, nj, ni, Ps)];
                }
                simga_nGi[imgAcc2(j0/stp, i0/stp, nj, ni)] = tmp;
            } // i0
        } //j0

        // Algorithm 2
        memcpy(cpyx, x, sizeof(double) * w * h);

        // 特異値分解
        for(l = 0 ; l < L ; l++){
            memset(win, 0, sizeof(double) * w * h);
            memset(newx, 0, sizeof(double) * w * h);
            memset(tmpx, x, sizeof(double) * w * h);

            for(j0 = 0 ; j0 < h ; j0 += stp){
                for(i0 = 0 ; i0 < w ; i0 += stp){

                    // Construct XGi
                    for(n = 0 ; n < K ; n++){
                        vi = Gi[imgAcc3(j0/stp, i0/stp, n, nj, ni, K)] % Ws;
                        vj = (Gi[imgAcc3(j0/stp, i0/stp, n, nj, ni, K)] - vi)/Ws;
                        vi = vi - Ws/2;
                        vj = vj - Ws/2;
                        for(j = 0 ; j < rtPs ; j++){
                            for(i = 0 ; i < rtPs ; i++){
                                //XGi[n * Ps + (j * rtPs + i)]
                                XGi[n * Ps + (j * rtPs + i)] = tmpx[(j0 + j + vj + h) % h * w + (i0 + i + vi + w) % w];
                            }
                        }
                    }
                    
                    double _sigmas = 0.0;
                    for(i = 0 ; i < Ps * K ; i++){
                        _sigmas += (YGi[i] - XGi[i]) * (YGi[i] - XGi[i]);
                    }
                    memcpy(tmpXGi, XGi, sizeof(double) * Ps * K);
                    // s -> sigma
                    dgesvd("All", "All", &Ps, &K, tmpXGi, &Ps, s, u, &Ps, vt, &K, work, &lwork, &info);
                    
                    // soft threshold => 
                    simga2_nGi[imgAcc2(j0/stp, i0/stp, nj, ni)] = sigma_nGi[imgAcc2(j0/stp, i0/stp, nj, ni)] * sigma_nGi[imgAcc2(j0/stp, i0/stp, nj, ni)];

                    // λ 
                    for(i = 0 ; i < Ps ; i++){
                        lambda_kGi[i] = s[i];
                    }
                    // equ(14)
                    for(i = 0 ;  i < Ps ; i++){
                        lambda_dash[i] = sqrt(lambda_kGi[i] - sigma2[imgAcc2(j0/stp, i0/stp, nj, ni)]);
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
                    memset(XGk, 0, sizeof(double) * Ps * K);
                    
                    for(j = 0 ; j < Ps ; j++){
                        for(i = 0 ; i < Ps ; i++){
                            us[imgAcc2(j, i, K, K)] = u[imgAcc(i, j, Ps, Ps)] * tmps[i];
                        }
                    }

                    for(j = 0 ; j < Ps ; j++){
                        for(i = 0 ; i < K ; i++){
                            for(n = 0 ; n < K ; n++){
                                XGk[Ps * i + j] += us[Ps * j + n] * vt[c * i + n];
                            }
                        }
                    }
                    // (15)
                    for(j = 0 ; j < rtPs ; j++){
                        for(i = 0 ; i < rtPs ; i++){
                            newx[(j0 + j) % h * w + (i0 + i) % w] += XGk[rtPs * j + i];
                            win[(j0 + j)%h * w + (i0 + i) % w]++;
                        }
                    }
                } // j0
            // } // i0
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
    free(lhat);
    free(uhat);
    free(Ax);
    free(newx);
    free(cpyx);
    free(tmpx);
    free(win);
    free(Gk);
    free(chk);
    free(err);
    free(ref);
    free(tgt);
    free(XGk);
    free(ZGk);
    free(cpyZGk);
    free(tmpZGk);
    free(wgt);
    free(us);
    free(tmps);
    free(s);
    free(u);
    free(vt);
    free(work);
}