// g++ -g -std=c++11 -O3 -march=native  inner_prod.cpp -o inner_prod
#include <stdio.h>
#include "utils.h"
#include <immintrin.h>
#include "intrin-wrapper.h"


//Naive
void compute_fn01(double* a, double* b, int N, double &c){
    double sum=0;
    for (long i = 0; i < N; i ++) {
        sum+=a[i]+b[i];
    }
    c = sum;
}
//Unroll 2, pipeline
void compute_fn02(double* a, double* b, int N, double &c){
    double sum1=0,sum2=0,temp=0;
    for (long i = 0; i < N/2-1; i ++) {
        sum1 += a[2*i] * b[2*i];
        sum2 += a[2*i+1] * b[2*i+1];

    }
    c=sum1+sum2;
}
//Unroll 2, pipeline with index optimization
void compute_fn03(double* a, double* b, int N, double &c){
    double sum1=0,sum2=0,temp1=0,temp2=0;
    for (long i = 0; i < N/2-1; i ++) {        
        temp1 = *(a + 0) * *(b + 0);        
        temp2 = *(a + 1) * *(b + 1);
        sum1 += temp1;sum2 += temp2;
        a += 2; b += 2;
    }
    c = sum1+sum2;
}

//Unroll 2, pipeline with index optimization and disentangle
void compute_fn04(double* a, double* b, int N, double &c){
    double sum1=0,sum2=0,temp1=0,temp2=0;
    for (long i = 0; i < N/2-1; i ++) {
        sum1 += temp1;
        temp1 = *(a + 0) * *(b + 0);
        sum2 += temp2;
        temp2 = *(a + 1) * *(b + 1);
        a += 2; b += 2;
    }
    c = sum1+sum2;
}

//Unroll 4, pipeline with index optimization and disentangle
void compute_fn05(double* a, double* b, int N, double &c){
    double sum1=0,sum2=0,sum3=0,sum4=0,temp1=0,temp2=0,temp3=0,temp4=0;
    for (long i = 0; i < N/4-1; i ++) {
        sum1 += temp1;
        temp1 = *(a + 0) * *(b + 0);
        sum2 += temp2;
        temp2 = *(a + 1) * *(b + 1);
        sum3 += temp3;
        temp3 = *(a + 2) * *(b + 2);
        sum4 += temp4;
        temp4 = *(a + 3) * *(b + 3);
        a += 4; b += 4;
    }
    c=sum1+sum2+sum3+sum4;
}



int main(int argc, char** argv){
    int iters=19;
    long n=16;
    double c,c_ref;
    printf("      Size       Time    Gflop/s\n");
    for (long iter = 0; iter < iters; iter++)
    {
        n=n*2;
        long NREPEATS = 1e9/n+1;
        double time;
        double flops,bandwidth;
        double* a = (double*) aligned_malloc(n*sizeof(double)); 
        double* b = (double*) aligned_malloc(n*sizeof(double)); 
        
        Timer t;
        for (long i = 0; i < n; i++) a[i] = drand48();
        for (long i = 0; i < n; i++) b[i] = drand48();
        t.tic();
        for (long rep = 0; rep < NREPEATS; rep++) {
            // Test function
            compute_fn01(a,b,n,c);   
        }        
        time=t.toc();
        flops=NREPEATS*n*2/time/1e9;
        printf("%10ld %10f %10f\n", n, time, flops);  
        aligned_free(a);
        aligned_free(b);
    }
    printf("Inner Product is %f\n",c); 
    

}
