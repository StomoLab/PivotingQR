#include <chrono>
#include <iostream>
#include <random>
#include <format>
#include "cblas.h"
#include "lapacke.h"

void pivoting_QR(double* A, const int M, const int N)
{
    auto A2 = new double[M*N];
    auto Q = new double[M*N];

    double one {1.0};

    for(int i=0; i<M*N; ++i){
        A2[i] = A[i];
    }

    //単位行列作成
    for( int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            if( i==j){
                Q[i + M*j] = {1.0};
            }
            else{
                Q[i + M*j] = {0.0};
            }
        }
    }

    auto dgeqpf_start = std::chrono::high_resolution_clock::now();

    auto jpiv = new int[N];
    auto work = new double[3*N];
    auto tau = new double[N];

    for(int j=0; j<N; ++j){
        jpiv[j] = j;
    }

    //最初に全縦ベクトルのノルムを調べる
    for(int j=0; j<N; ++j){
        work[j] = cblas_dnrm2( M, &A[0 + M*j], 1);
        work[N+j] = work[j];
    }

    for(int i=0; i<std::min(M,N); ++i){
        //i+1 ~ N までの中でノルムが大きいものを調べる
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
            LAPACKE_dlarfg( (M-i), &A[i + M*i], &A[i+1 + M*i], 1, &tau[i]);
        }
        if( i < (N-1)) {
            //ハウスホルダーベクトルを後続にかける
            double aii = A[i + M*i];
            A[i + M*i] = 1.0;
            LAPACKE_dlarfx( LAPACK_COL_MAJOR,'L', M-i, N-(i+1), &A[i + M*i], tau[i], &A[i + M*(i+1)], M, &work[2*N]);
            A[i + M*i] = aii;
        }

        //更新後のノルム更新
        for(int j=i+1; j<N; ++j){
            //これが重い
            //work[j] = cblas_dnrm2( M-(i+1), &A[(i+1) + (M*j)], 1);
        }
    }
    //QR分解終了
    auto dgeqpf_end = std::chrono::high_resolution_clock::now();
    std::cout << "dgeqpf:" << std::chrono::duration_cast<std::chrono::microseconds>(dgeqpf_end - dgeqpf_start).count() << std::endl;


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

    auto inorm = cblas_dnrm2( M*N, &I[0], 1);
    auto dnorm = cblas_dnrm2( M*N, &A2[0], 1);

    std::cout << "直交性:" << inorm << std::endl;
    std::cout << "残差:" << dnorm << std::endl;

    delete[] A2;
    delete[] jpiv;
    delete[] work;
    delete[] tau;

    delete [] jpiv2;
    delete [] Q;
    delete [] I;
}

int main() {
    int M  {1024};
    int N  {1024};
    int nb {64};

    auto A1 = new double[M*N];
    auto A2 = new double[M*N];
    auto A3 = new double[M*N];

    int one {1};

    std::random_device rd;
    std::mt19937_64 gen( rd() );
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for(int i=0; i<M*N; ++i){
        A1[i] = dis(gen);
    }

    for(int i=0; i<M*N; ++i){
        A2[i] = A1[i];
    }

    for (int i=0; i<M*N; ++i) {
        A3[i] = A1[i];
    }

    int info {0};
    //ブロック化ルーチン



    auto jpiv = new int[N];

    auto work = new double[3*N];
    auto tau = new double[N];

    for(int j=0; j<N; ++j){
        jpiv[j] = {0};
    }

    auto dgeqpf_start = std::chrono::high_resolution_clock::now();
    LAPACKE_dgeqpf(LAPACK_COL_MAJOR, M, N, A1, M, jpiv, tau);
    auto dgeqpf_end = std::chrono::high_resolution_clock::now();
    std::cout << "dgeqpf:" << std::chrono::duration_cast<std::chrono::microseconds>(dgeqpf_end - dgeqpf_start).count() << std::endl;

    auto dgeqp3_start = std::chrono::high_resolution_clock::now();
    LAPACKE_dgeqp3(LAPACK_COL_MAJOR, M, N, A2, M, jpiv, tau);
    auto dgeqp3_end = std::chrono::high_resolution_clock::now();

    std::cout << "dgeqp3:" << std::chrono::duration_cast<std::chrono::microseconds>(dgeqp3_end - dgeqp3_start).count() << std::endl;


    //LAPACKE_dgeqp3(LAPACK_COL_MAJOR, M, N, A,M, jpiv, tau);
    //pivoting_QR(A1, M, N);



    for( int i=0; i<M; ++i){
        if( A1[i+M*i] != A2[i+M*i]) {
            std::cout << std::format("{0}:{1},{2}\n",i+M*i,A1[i + M * i], A2[i + M * i]);
        }
    }

    // A2 = AP を作成
    auto jpiv2 = new int[N];
    for(int j=0; j<N;++j){
        jpiv2[j] = j;
    }

    for(int j=0; j<N; ++j){
        for(int jj=0; jj<N; ++jj){
            //fortranなのでindexが1始まり
            if(jpiv[j]-1 == jpiv2[jj]){
                cblas_dswap(M, &A3[0 + M * j], 1, &A3[0 + M * jj], 1);
                std::swap(jpiv2[j],jpiv2[jj]);
            }
        }
    }

    //Q行列作成
    auto Q = new double[M*N];

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

    for(int j = N-1; j>=0; --j){
        double ajj = A2[j + M*j];
        A2[j + M*j] = one;
        LAPACKE_dlarfx( LAPACK_COL_MAJOR,'L', M-j, N-j, &A2[j + M*j], tau[j], &Q[j+ M*j], M, &work[2*N]);
        A2[j + M*j] = ajj;
    }

    //R作成
    for(int i=0; i<M; ++i){
        for(int j=0; j<N; ++j){
            if( i > j ){
                A2[i + M*j] = 0.0;
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

    // Q*R - A3(ピボット済み)
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, M, one, &Q[0], M, &A2[0], M, -one, &A3[0], M);

    auto inorm = cblas_dnrm2( M*N, &I[0], 1);
    auto dnorm = cblas_dnrm2( M*N, &A3[0], 1);

    std::cout << "直交性:" << inorm << std::endl;
    std::cout << "残差:" << dnorm << std::endl;



    delete[] A1;
    delete[] A2;
    delete[] A3;
    delete[] jpiv;
    delete[] work;
    delete[] tau;

    delete [] jpiv2;
    delete [] Q;
    delete [] I;
    return 0;
}
