#include <iostream>
#include <random>
#include "cblas.h"
#include "lapacke.h"

int main() {
    int M = 4096;
    int N = 4096;
    auto A = new double[M*N];
    auto A2 = new double[M*N];

    auto Q = new double[M*N];

    int one = 1;

    std::random_device rd;
    std::mt19937_64 gen( rd() );
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for(int i=0; i<M*N; ++i){
        A[i] = dis(gen);
    }

    for(int i=0; i<M*N; ++i){
        A2[i] = A[i];
    }

    //単位行列作成
    for( int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            if( i==j){
                Q[i + M*j] = 1.0;
            }
            else{
                Q[i + M*j] = 0.0;
            }
        }
    }

    auto jpiv = new int[N];

    auto work = new double[3*N];
    auto tau = new double[N];

    for(int j=0; j<N; ++j){
        jpiv[j] = j;
    }

    for(int i=0; i<N; ++i){
        work[i] = cblas_dnrm2( M-i, &A[i + M*i], 1);
    }

    for(int i=0; i<std::min(M,N); ++i){

        //std::cout << "Pivot:" << i << std::endl;

        int pvt = i + cblas_idamax( N-i, &work[i], 1);

        //ピボット
        if( pvt != i ) {
            cblas_dswap(M, &A[0 + M * pvt], 1, &A[0 + M * i], 1);
            std::swap(jpiv[pvt], jpiv[i]);
            work[pvt] = work[i];
            work[N + pvt] = work[N + i];
        }

        if (i < M) {
            //ハウスホルダーを作る
            LAPACKE_dlarfg( (M - i), &A[i + M * i], &A[i + 1 + M * i], one, &tau[i]);
            //std::cout << "LARFG:" << i << std::endl;
        }
        if( i < (N-1)) {
            //ハウスホルダーベクトルを後続にかける
            double aii = A[i + M*i];
            A[i + M*i] = one;
            LAPACKE_dlarfx( LAPACK_COL_MAJOR,'L', M-i, N-(i+1), &A[i + M*i], tau[i], &A[i + M*(i+1)], M, &work[2*N]);
            A[i + M*i] = aii;
            //std::cout << "LARFT:" << i << std::endl;
        }
    }
    //QR分解終了

    // A2 = AP を作成
    auto jpiv2 = new int[N];
    for(int j=0; j<N;++j){
        jpiv2[j] = j;
    }

    for(int j=0; j<N; ++j){
        for(int jj=0; jj<N; ++jj){
            if(jpiv[j] == jpiv2[jj]){
                cblas_dswap(M, &A2[0 + M * j], 1, &A2[0 + M * jj], 1);
                std::swap(jpiv2[j],jpiv2[jj]);
            }
        }
    }

    //Q行列作成
    for(int j = N-1; j>=0; --j){
        double ajj = A[j + M*j];
        A[j + M*j] = one;
        LAPACKE_dlarfx( LAPACK_COL_MAJOR,'L', M-j, N-j, &A[j + M*j], tau[j], &Q[j+ M*j], M, &work[2*N]);
        A[j + M*j] = ajj;
    }

    //R作成
    for(int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            if( i > j ){
                A[i + M*j] = 0.0;
            }
        }
    }

    auto I = new double[M*N];

    for( int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            if( i==j){
                I[i + M*j] = 1.0;
            }
            else{
                I[i + M*j] = 0.0;
            }
        }
    }
    // Q*Qt - I
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans, M, N, M, one, &Q[0], M, &Q[0], M, -one, &I[0], M);

    // Q*R - A2(ピボット済み)
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, M, one, &Q[0], M, &A[0], M, -one, &A2[0], M);


    //対角要素確認用
    /*
    for( int i=0; i<M; ++i){
        std::cout << A[i+M*i] << std::endl;
    }
     */

    /*
    for( int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            std::cout << A2[i + M*j] << ',';
        }
        std::cout << std::endl;
    }
*/
    auto inorm = cblas_dnrm2( M*N, &I[0], 1);
    auto dnorm = cblas_dnrm2( M*N, &A2[0], 1);

    std::cout << "直交性:" << inorm << std::endl;
    std::cout << "残差:" << dnorm << std::endl;

    delete[] A;
    delete[] A2;
    delete[] jpiv;
    delete[] work;
    delete[] tau;

    delete [] jpiv2;
    delete [] Q;
    delete [] I;
    return 0;
}
