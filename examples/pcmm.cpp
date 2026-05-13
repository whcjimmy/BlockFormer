#include "HEaaNTimer.hpp"
#include "examples.hpp"
#include <HEaaN/Bootstrapper.hpp> // CPU MODE
#include <HEaaN/Ciphertext.hpp>
#include <HEaaN/Context.hpp>
#include <HEaaN/EnDecoder.hpp>
#include <HEaaN/Message.hpp>
#include <HEaaN/SecretKey.hpp>
#include <HEaaN/device/CudaTools.hpp>
#include <HEaaN/device/Device.hpp>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip> 
// #include "/usr/local/cuda/include/cuda_runtime.h"

extern double total_time;
double vm, vm2, rss, rss2;

int main(int argc, char *argv[]) {
    HEaaN::HEaaNTimer timer(true);
    // You can use other bootstrappable parameter instead of FGb.
    // See 'include/HEaaN/ParameterPreset.hpp' for more details.
    HEaaN::ParameterPreset preset = HEaaN::ParameterPreset::FGb;
    HEaaN::Context context = makeContext(preset);
    if (!HEaaN::isBootstrappableParameter(context)) {
        std::cout << "Bootstrap is not available for parameter "
                  << presetNamer(preset) << std::endl;
        return -1;
    }

    std::cout << "Parameter : " << presetNamer(preset) << std::endl
              << std::endl;

    const auto log_slots = getLogFullSlots(context);
    const auto num_slots = UINT64_C(1) << log_slots;

    HEaaN::SecretKey sk(context);
    HEaaN::KeyPack pack(context);
    HEaaN::KeyGenerator keygen(context, sk, pack);
    HEaaN::EnDecoder edcoder(context);

    std::cout << "Generate encryption key ... " << std::endl;
    keygen.genEncryptionKey();
    keygen.genRotKeysForBootstrap(log_slots); // CPU MODE
    // HEaaN::makeBootstrappable(context); // GPU MODE
    std::cout << "done" << std::endl << std::endl;

    std::cout << "Generate commonly used keys (mult key, rotation keys, "
                 "conjugation key) ... "
              << std::endl;
    keygen.genCommonKeys();
    std::cout << "done" << std::endl << std::endl;

    HEaaN::Encryptor enc(context);
    HEaaN::Decryptor dec(context);

    /*
    HomEvaluator constructor pre-compute the constants for bootstrapping.
    */
    std::cout << "Generate HomEvaluator (including pre-computing constants for "
                 "bootstrapping) ..."
              << std::endl;
    timer.start("* ");
    HEaaN::HomEvaluator eval(context, pack);
    HEaaN::Bootstrapper boot(eval, log_slots); // CPU MODE
    timer.end();
    timer.print();

    // PCMM 
    // matrix A (m x n), matrix B (n x k), matrix O (m x k)
    int m = atoi(argv[1]); 
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int alpha = atoi(argv[4]);
    int lvl = atoi(argv[5]);
    int is_gpu = atoi(argv[6]) == 1 ? true : false;
    int t = n/alpha;
    int u = k/alpha;

    int out_size = 8;

    std::cout << "m, n, k, alpha " << m << " " << n << " " << k << " " << alpha << std::endl;
    std::cout << "t: " << t << " " << "u: " << u << std::endl;

    std::cout << "=========\n";
    if(is_gpu) {
        std::cout << "GPU MODE\n";
    }else {
        std::cout << "CPU MODE\n";
    }
    std::cout << "=========\n";

    std::vector<std::vector<double>> matrixA(m, std::vector<double>(n, 0.0));
    std::vector<std::vector<double>> matrixB(n, std::vector<double>(k, 0.0));
    std::vector<std::vector<double>> matrixO(m, std::vector<double>(k, 0.0));
    std::vector<std::vector<double>> matrixOC(m, std::vector<double>(k, 0.0));

    srand(time(NULL));

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            matrixA[i][j] = (double)rand() / RAND_MAX;
            // matrixA[i][j] = 0.001 * randNum() * (i + j + 0.1);
            // matrixA[i][j] = 0.1 * (i + j + 0.1);
            // matrixA[i][j] = 1;
        }
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < k; j++) {
            matrixB[i][j] = (double)rand() / RAND_MAX;
            // matrixB[i][j] = 0.001 * randNum() * (i + j + 0.1);
            // matrixB[i][j] = 0.1 * (i + j + 0.1);
            // matrixB[i][j] = 1;
        }
    }

    std::cout << "=== Preprocessing ===" << std::endl;

    HEaaN::Message zero(log_slots);
    fillZero(zero);

    std::vector<HEaaN::Ciphertext> cipherA(t, HEaaN::Ciphertext(context));
    for(int aa = 0; aa < t; aa++) {
        HEaaN::Message msg(log_slots);
        fillZero(msg);
#pragma omp parallel for collapse(2)
        for(int ii = 0; ii < m; ii++) {
            for(int jj = 0; jj < alpha; jj++) {
                msg[ii * alpha + jj].real(matrixA[ii][aa * alpha + jj]);
            }
        }
        enc.encrypt(msg, pack, cipherA[aa]);
        cipherA[aa].setLevel(lvl);
    }

    std::vector<std::vector<HEaaN::Message>> msgB(t, std::vector<HEaaN::Message>(k, HEaaN::Message(log_slots)));
#pragma omp parallel for collapse(2)
    for(int aa = 0; aa < t; aa++) {
        for(int kk = 0; kk < k; kk++) {
            HEaaN::Message msg(log_slots);
            fillZero(msg);
            for(int ii = 0; ii < m; ii++) {
                for(int jj = 0; jj < alpha; jj++) {
                    msg[ii * alpha + jj].real(matrixB[aa * alpha + jj][kk]);
                }
            }
            msgB[aa][kk] = msg;
        }
    }

    std::cout << "=== MM computation ===" << std::endl;
    
    std::vector<HEaaN::Ciphertext> cipherO(u, HEaaN::Ciphertext(context));
    for(int bb = 0; bb < u; bb++) {
        enc.encrypt(zero, pack, cipherO[bb]);
        cipherO[bb].setLevel(lvl - 2);
    }

    HEaaN::Message filter_msg(log_slots);
    fillZero(filter_msg);
    int tt = 0;
    while(tt < num_slots) {
        filter_msg[tt].real(1.0);
        filter_msg[tt].imag(0.0);
        tt += alpha;
    }

    if(is_gpu) {
        std::cout << "Transfer Data to GPU" << std::endl;

        filter_msg.to(HEaaN::DeviceType::GPU);

        for(int i = 0; i < t; i++) {
            cipherA[i].to(HEaaN::DeviceType::GPU);
        }

        for(int i = 0; i < t; i++) {
            for(int j = 0; j < k; j++) {
                msgB[i][j].to(HEaaN::DeviceType::GPU);
            }
        }

        for(int i = 0; i < u; i++) {
            cipherO[i].to(HEaaN::DeviceType::GPU);
        }

        pack.to(HEaaN::DeviceType::GPU);
    }

    timer.start("MM computation");
#pragma omp parallel for collapse(2) if(!is_gpu)
    for(int bb = 0; bb < u; bb++) {
        for(int dd = 0; dd < alpha; dd++) {
            HEaaN::Ciphertext res_cipher(context), tmp_cipher(context);

            for(int aa = 0; aa < t; aa++) {
                eval.mult(cipherA[aa], msgB[aa][bb * alpha + dd], tmp_cipher); // 1 lvl
                if(aa == 0) {
                    res_cipher = tmp_cipher;
                }else{
                    eval.add(res_cipher, tmp_cipher, res_cipher);
                }
            }

            // Rotate and add
            tmp_cipher = res_cipher;
            for(int p = log2(alpha) - 1; p >= 0; p--) {
                eval.leftRotate(tmp_cipher, pow(2, p), tmp_cipher);
                eval.add(res_cipher, tmp_cipher, res_cipher);
                tmp_cipher = res_cipher;
            }

            eval.mult(res_cipher, filter_msg, res_cipher); // 1 lvl
            eval.rightRotate(res_cipher, dd, res_cipher);

#pragma omp critical
            {
                eval.add(cipherO[bb], res_cipher, cipherO[bb]);
            }
        }
    }
    timer.end();
    timer.print();

    if(is_gpu) {
        std::cout << "Transfer Data back to CPU" << std::endl;

        filter_msg.to(HEaaN::DeviceType::CPU);

        for(int i = 0; i < t; i++) {
            cipherA[i].to(HEaaN::DeviceType::CPU);
        }

        for(int i = 0; i < t; i++) {
            for(int j = 0; j < k; j++) {
                msgB[i][j].to(HEaaN::DeviceType::CPU);
            }
        }

        for(int i = 0; i < u; i++) {
            cipherO[i].to(HEaaN::DeviceType::CPU);
        }

        pack.to(HEaaN::DeviceType::CPU);
    }


    std::cout << "=== Plaintext Result ===" << std::endl;
    for(int mm = 0; mm < m; mm++) {
        for(int kk = 0; kk < k; kk++) {
            double tmp = 0;
            for(int nn = 0; nn < n; nn++) {
                tmp += matrixA[mm][nn] * matrixB[nn][kk];
            }
            matrixO[mm][kk] = tmp;
        }
    }

    for(int mm = 0; mm < out_size; mm++) {
        for(int kk = 0; kk < out_size; kk++) {
            std::cout << std::fixed << std::setprecision(2) << matrixO[mm][kk] << " ";
        }
        std::cout << std::endl;
    }


    std::cout << "=== Encrypted Result ===" << std::endl;
    for(int bb = 0; bb < u; bb++) {
        HEaaN::Message result_msg;
        dec.decrypt(cipherO[bb], sk, result_msg);
        for(int mm = 0; mm < m; mm++) {
            for(int dd = 0; dd < alpha; dd++) {
                matrixOC[mm][bb * alpha + dd] = result_msg[mm * alpha + dd].real();
            }
        }
    }

    for(int i = 0; i < out_size; i++) {
        for(int j = 0; j < out_size; j++) {
            std::cout << std::fixed << std::setprecision(2) << matrixOC[i][j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
