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

void QKT(HEaaN::Context &context, HEaaN::HomEvaluator &eval, HEaaN::KeyPack &pack, 
         std::vector<HEaaN::Ciphertext> &cipherA, 
         std::vector<HEaaN::Ciphertext> &cipherB, 
         std::vector<std::vector<HEaaN::Ciphertext>> &cipherO, 
         int m, int n, int num_inputs, int num_QKT_head_cts, int num_QKT_cts, int head_dim, int log_slots, int num_slots, int is_gpu) {
    HEaaN::HEaaNTimer timer(true);

    int num_filter = (n <= head_dim) ? 1 : 2;
    std::cout << "num_filter: " << num_filter << std::endl;
    std::vector<std::vector<HEaaN::Message>> filter_msg_1_vec(m, std::vector<HEaaN::Message>(num_filter, HEaaN::Message(log_slots)));
    for(int ii = 0; ii < m; ii++) {
        for(int jj = 0; jj < num_filter; jj++) {
            fillZero(filter_msg_1_vec[ii][jj]);
        }

        int len = (n <= head_dim) ? n : head_dim;
        int f = 0;
        for(int i = 0; i < n; i+=len) {
            int start = ii * n + i;
            int end = start + len;
            for(int jj = start; jj < end; jj++) {
                filter_msg_1_vec[ii][f][jj].real(1.0);
            }
            f = (f == 0) ? 1 : 0;
        }
    }

    HEaaN::Message filter_msg_2(log_slots);
    fillZero(filter_msg_2);
    int t = 0;
    while(t < num_slots) {
        filter_msg_2[t].real(1.0);
        filter_msg_2[t].imag(0.0);
        t += n;
    }
    
    if(is_gpu) {
        std::cout << "Transfer Data to GPU" << std::endl;

        for(int i = 0; i < m; i++) {
            for(int j = 0; j < num_filter; j++) {
                filter_msg_1_vec[i][j].to(HEaaN::DeviceType::GPU);
            }
        }
        filter_msg_2.to(HEaaN::DeviceType::GPU);

        for(int input = 0; input < num_inputs; input++) {
            cipherA[input].to(HEaaN::DeviceType::GPU);
            cipherB[input].to(HEaaN::DeviceType::GPU);
        }

        for(int i = 0; i < num_QKT_head_cts; i++) {
            for(int j = 0; j < num_QKT_cts; j++) {
                cipherO[i][j].to(HEaaN::DeviceType::GPU);
            }
        }

        pack.to(HEaaN::DeviceType::GPU);
    }

    timer.start("Q X KT computation");
    for(int filter = 0; filter < num_filter; filter++) {
#pragma omp parallel for collapse(2) if(!is_gpu)
        for(int input = 0; input < num_inputs; input++) {
            for(int mm = 0; mm < m; mm++) {
                HEaaN::Ciphertext filtered_B_cipher(context);
                eval.mult(cipherB[input], filter_msg_1_vec[mm][filter], filtered_B_cipher); // 1 LVL
                eval.leftRotate(filtered_B_cipher, mm * n, filtered_B_cipher);

                HEaaN::Ciphertext res_cipher(context), tmp_cipher(context);
                tmp_cipher = filtered_B_cipher;
                for(int ii = 0; ii < log2(m); ii++) {
                    eval.rightRotate(tmp_cipher, n * pow(2, ii), tmp_cipher);
                    eval.add(filtered_B_cipher, tmp_cipher, filtered_B_cipher);
                    tmp_cipher = filtered_B_cipher;
                }

                eval.mult(cipherA[input], filtered_B_cipher, res_cipher); // 1 LVL

                tmp_cipher = res_cipher;
                int len = (n <= head_dim) ? n : head_dim;
                for(int ii = log2(len) - 1; ii >= 0; ii--) {
                    eval.leftRotate(tmp_cipher, pow(2, ii), tmp_cipher);
                    eval.add(res_cipher, tmp_cipher, res_cipher);
                    tmp_cipher = res_cipher;
                }

                eval.mult(res_cipher, filter_msg_2, res_cipher); // 1 LVL
                eval.rightRotate(res_cipher, mm%n, res_cipher);

#pragma omp critical
                {
                    int head_idx;
                    if(n <= head_dim) {
                        head_idx = std::floor(input * n / head_dim);
                    }else {
                        head_idx = input * 2 + filter; // 2 = 128 / 64
                    }
                    int QKT_ct_idx = int(mm / n);
                    eval.add(cipherO[head_idx][QKT_ct_idx], res_cipher, cipherO[head_idx][QKT_ct_idx]);
                }
            }
        }
    }
    timer.end();
    timer.print();

    if(is_gpu) {
        std::cout << "Transfer Data back to CPU" << std::endl;

        for(int i = 0; i < m; i++) {
            for(int j = 0; j < num_filter; j++) {
                filter_msg_1_vec[i][j].to(HEaaN::DeviceType::GPU);
            }
        }
        filter_msg_2.to(HEaaN::DeviceType::CPU);

        for(int input = 0; input < num_inputs; input++) {
            cipherA[input].to(HEaaN::DeviceType::CPU);
            cipherB[input].to(HEaaN::DeviceType::CPU);
        }

        for(int i = 0; i < num_QKT_head_cts; i++) {
            for(int j = 0; j < num_QKT_cts; j++) {
                cipherO[i][j].to(HEaaN::DeviceType::CPU);
            }
        }

        pack.to(HEaaN::DeviceType::CPU);
    }
}

void QKTV(HEaaN::Context &context, HEaaN::HomEvaluator &eval, HEaaN::KeyPack &pack,
          std::vector<std::vector<HEaaN::Ciphertext>> &cipherO, 
          std::vector<HEaaN::Ciphertext> &cipherC, 
          std::vector<HEaaN::Ciphertext> &cipherO2,
          int m, int n, int num_inputs, int num_QKT_head_cts, int num_QKT_cts, int head_dim, int log_slots, int num_slots, int is_gpu) {
    HEaaN::HEaaNTimer timer(true);

    // filter 1
    // Diag Transpose matrix V
    int minima = std::min(m, n);
    std::vector<std::vector<HEaaN::Message>> filter_msg_1_vec(minima, std::vector<HEaaN::Message>(2, HEaaN::Message(log_slots)));

    for(int i = 0; i < minima; i++) {
        for(int j = 0; j < 2; j++) {
            fillZero(filter_msg_1_vec[i][j]);
        }
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            int diff = abs(i % minima - j % minima);
            if(i % minima > j % minima) {
                filter_msg_1_vec[diff][0][i * n + j].real(1.0); // Lower
            }else if (i % minima < j % minima) {
                filter_msg_1_vec[diff][1][i * n + j].real(1.0); // Upper
            }else if (i % minima == j % minima) {
                filter_msg_1_vec[diff][0][i * n + j].real(1.0); // Diagonal
            }
        }
    }

    // filter 2
    // NOTE: m = num_col * num_filter
    int num_col = (n <= head_dim) ? n : head_dim;
    int num_filter = (n <= head_dim) ? 1 : 2;
    std::cout << "num_filter: " << num_filter << std::endl;
    std::vector<std::vector<std::vector<HEaaN::Message>>> filter_msg_2_vec(num_col, 
                                  std::vector<std::vector<HEaaN::Message>>(num_QKT_cts,
                                               std::vector<HEaaN::Message>(num_filter, HEaaN::Message(log_slots))));
    for(int col = 0; col < num_col; col++) {
        for(int ct = 0; ct < num_QKT_cts; ct++) {
            for(int filter = 0; filter < num_filter; filter++) {
                fillZero(filter_msg_2_vec[col][ct][filter]);
            }
        }
    }

    for(int col = 0; col < num_col; col++) {
        for(int ct = 0; ct < num_QKT_cts; ct++) {
            for(int filter = 0; filter < num_filter; filter++) {
                int start = col * n + ct * n * n + filter * head_dim * n;
                int end = start + n;
                for(int i = start; i < end; i++) {
                    filter_msg_2_vec[col][ct][filter][i].real(1.0);
                }
            }
        }
    }

    // filter 4
    HEaaN::Message filter_msg_4(log_slots);
    fillZero(filter_msg_4);
    int t = 0;
    while(t < num_slots) {
        filter_msg_4[t].real(1.0);
        filter_msg_4[t].imag(0.0);
        t += n;
    }

    if(is_gpu) {
        std::cout << "Transfer Data to GPU" << std::endl;

        for(int i = 0; i < minima; i++) {
            for(int j = 0; j < 2; j++) {
                filter_msg_1_vec[i][j].to(HEaaN::DeviceType::GPU);
            }
        }
        for(int col = 0; col < num_col; col++) {
            for(int ct = 0; ct < num_QKT_cts; ct++) {
                for(int filter = 0; filter < num_filter; filter++) {
                    filter_msg_2_vec[col][ct][filter].to(HEaaN::DeviceType::GPU);
                }
            }
        }
        filter_msg_4.to(HEaaN::DeviceType::GPU);

        for(int i = 0; i < num_QKT_head_cts; i++) {
            for(int j = 0; j < num_QKT_cts; j++) {
                cipherO[i][j].to(HEaaN::DeviceType::GPU);
            }
        }

        for(int i = 0; i < num_inputs; i++) {
            cipherC[i].to(HEaaN::DeviceType::GPU);
            cipherO2[i].to(HEaaN::DeviceType::GPU);
        }

        pack.to(HEaaN::DeviceType::GPU);
    }

    timer.start("Q X KTV computation");
    // Diag Transpose Matrix V. Cost 1 LVL in total
    for(int input = 0; input < num_inputs; input++) {
        HEaaN::Ciphertext res_cipher(context);

        int minima = std::min(m, n);
        eval.mult(cipherC[input], filter_msg_1_vec[0][0], res_cipher);
#pragma omp parallel for if(!is_gpu)
        for(int i = 1; i < minima; i++) {
            HEaaN::Ciphertext filtered_C_cipher_1(context), filtered_C_cipher_2(context);

            eval.mult(cipherC[input], filter_msg_1_vec[i][0], filtered_C_cipher_1);
            eval.mult(cipherC[input], filter_msg_1_vec[i][1], filtered_C_cipher_2);
            eval.leftRotate(filtered_C_cipher_1, n * i - i, filtered_C_cipher_1);
            eval.rightRotate(filtered_C_cipher_2, n * i - i, filtered_C_cipher_2);
            eval.add(filtered_C_cipher_1, filtered_C_cipher_2, filtered_C_cipher_2);
#pragma omp critical
            {
                eval.add(filtered_C_cipher_2, res_cipher, res_cipher);
            }
        }
        cipherC[input] = res_cipher;
    }

    // Calculation of S(QK^T)V
    for(int filter = 0; filter < num_filter; filter++) {
#pragma omp parallel for collapse(2) if(!is_gpu)
        for(int input = 0; input < num_inputs; input++) {
            for(int col = 0; col < num_col; col++) {
                for(int ct = 0; ct < num_QKT_cts; ct++) {
                    HEaaN::Ciphertext res_cipher(context), tmp_cipher(context);
                    HEaaN::Ciphertext filtered_C_cipher(context);

                    eval.mult(cipherC[input], filter_msg_2_vec[col][ct][filter], filtered_C_cipher); // 1 LVL
                    eval.leftRotate(filtered_C_cipher, (col + ct * n) * n, filtered_C_cipher);

                    tmp_cipher = filtered_C_cipher;
                    for(int ii = 0; ii < log2(m); ii++) {
                        eval.rightRotate(tmp_cipher, n * pow(2, ii), tmp_cipher);
                        eval.add(filtered_C_cipher, tmp_cipher, filtered_C_cipher);
                        tmp_cipher = filtered_C_cipher;
                    }

                    int input_idx;
                    if(n < head_dim) {
                        input_idx = int(input / (head_dim / n));
                    }else if (n == head_dim){
                        input_idx = input;
                    }else {
                        input_idx = input * 2 + filter;
                    }
                    eval.mult(cipherO[input_idx][ct], filtered_C_cipher, res_cipher); // 1 LVL

                    tmp_cipher = res_cipher;
                    int len = (n <= head_dim) ? n : m;
                    for(int ii = log2(len) - 1; ii >= 0; ii--) {
                        eval.leftRotate(tmp_cipher, pow(2, ii), tmp_cipher);
                        eval.add(res_cipher, tmp_cipher, res_cipher);
                        tmp_cipher = res_cipher;
                    }

                    eval.mult(res_cipher, filter_msg_4, res_cipher); // 1 LVL
                    int rot_len = (n <= head_dim) ? col % n : col % n + filter * head_dim;
                    eval.rightRotate(res_cipher, rot_len, res_cipher);

#pragma omp critical
                    {
                        eval.add(cipherO2[input], res_cipher, cipherO2[input]);
                    }
                }
            }
        }
    }

    timer.end();
    timer.print();

    if(is_gpu) {
        std::cout << "Transfer Data back to CPU" << std::endl;

        for(int i = 0; i < minima; i++) {
            for(int j = 0; j < 2; j++) {
                filter_msg_1_vec[i][j].to(HEaaN::DeviceType::CPU);
            }
        }
        for(int col = 0; col < num_col; col++) {
            for(int ct = 0; ct < num_QKT_cts; ct++) {
                for(int filter = 0; filter < num_filter; filter++) {
                    filter_msg_2_vec[col][ct][filter].to(HEaaN::DeviceType::CPU);
                }
            }
        }
        filter_msg_4.to(HEaaN::DeviceType::CPU);

        for(int i = 0; i < num_QKT_head_cts; i++) {
            for(int j = 0; j < num_QKT_cts; j++) {
                cipherO[i][j].to(HEaaN::DeviceType::GPU);
            }
        }

        for(int i = 0; i < num_inputs; i++) {
            cipherC[i].to(HEaaN::DeviceType::CPU);
            cipherO2[i].to(HEaaN::DeviceType::CPU);
        }

        pack.to(HEaaN::DeviceType::CPU);
    }
}

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

    // MM 
    int m = atoi(argv[1]); // input length
    int n = atoi(argv[2]); // a.k.a. alpha
    int is_gpu = atoi(argv[3]) == 1 ? true : false;
    int head_dim = 64;
    int emb_dim = 768;
    int num_heads = emb_dim / head_dim; // 768/64=12
    int num_inputs = num_heads * head_dim / n;
    int num_QKT_head_cts = (n <= m) ? num_heads : num_heads * head_dim / m ; // 2 = \lceil d/head_dim \rceil / \lceil d/m \rceil
    int num_QKT_cts = (n <= head_dim) ? (m / n) : 1; // 1 = \lceil m^2 / S \rceil

    // int out_size = std::min(head_dim, n);
    int out_size = 8;

    std::cout << std::endl;
    std::cout << "m, n: " << m << " " << n << std::endl;
    std::cout << "num_inputs: " << num_inputs << std::endl;
    std::cout << "num_QKT_head_cts: " << num_QKT_head_cts << std::endl;
    std::cout << "num_QKT_cts: " << num_QKT_cts << std::endl;
    std::cout << std::endl;

    std::cout << "=========\n";
    if(is_gpu) {
        std::cout << "GPU MODE\n";
    }else {
        std::cout << "CPU MODE\n";
    }
    std::cout << "=========\n";

    std::vector<std::vector<double>> matrixA(m, std::vector<double>(n, 0.0));   // Q 
    std::vector<std::vector<double>> matrixB(m, std::vector<double>(n, 0.0));   // K
    std::vector<std::vector<double>> matrixC(m, std::vector<double>(n, 0.0));   // V
    std::vector<std::vector<double>> matrixO(m, std::vector<double>(m, 0.0));   // QKT Output
    std::vector<std::vector<double>> matrixOC(m, std::vector<double>(m, 0.0));  //
    std::vector<std::vector<double>> matrixO2(m, std::vector<double>(n, 0.0));  // QKTV Output
    std::vector<std::vector<double>> matrixOC2(m, std::vector<double>(n, 0.0)); //

    srand(time(NULL));

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            // matrixA[i][j] = 0.001 * (i + j + 0.1);
            // matrixB[i][j] = 0.001 * (i + j + 0.2);
            // matrixC[i][j] = 0.01 * (i + j + 0.3);
            // matrixA[i][j] = 0.1;
            // matrixB[i][j] = 0.1;
            // matrixC[i][j] = 0.1;
            matrixA[i][j] = (double)rand() / RAND_MAX;
            matrixB[i][j] = (double)rand() / RAND_MAX;
            matrixC[i][j] = (double)rand() / RAND_MAX;
        }
    }

    std::cout << "=== Preprocessing ===" << std::endl;

    HEaaN::Message zero(log_slots);
    fillZero(zero);

    std::vector<HEaaN::Ciphertext> cipherA(num_inputs, HEaaN::Ciphertext(context));
    std::vector<HEaaN::Ciphertext> cipherB(num_inputs, HEaaN::Ciphertext(context));
    std::vector<HEaaN::Ciphertext> cipherC(num_inputs, HEaaN::Ciphertext(context));

    HEaaN::Message msgA(log_slots), msgB(log_slots), msgC(log_slots);
    fillZero(msgA);
    fillZero(msgB);
    fillZero(msgC);
    for(int ii = 0; ii < m; ii++) {
        for(int jj = 0; jj < n; jj++) {
            msgA[ii * n + jj].real(matrixA[ii][jj]);
            msgB[ii * n + jj].real(matrixB[ii][jj]);
            msgC[ii * n + jj].real(matrixC[ii][jj]);
        }
    }

    for(int ii = 0; ii < num_inputs; ii++) {
        enc.encrypt(msgA, pack, cipherA[ii]);
        enc.encrypt(msgB, pack, cipherB[ii]);
        enc.encrypt(msgC, pack, cipherC[ii]);

        cipherA[ii].setLevel(9);
        cipherB[ii].setLevel(10);
        cipherC[ii].setLevel(7);
    }

    std::cout << "=== QKT computation (SSBMM) ==" << std::endl;

    std::vector<std::vector<HEaaN::Ciphertext>> cipherO(num_QKT_head_cts, std::vector<HEaaN::Ciphertext>(num_QKT_cts, HEaaN::Ciphertext(context)));
    for(int ii = 0; ii < num_QKT_head_cts; ii++){
        for(int jj = 0; jj < num_QKT_cts; jj++) {
            enc.encrypt(zero, pack, cipherO[ii][jj]);
            cipherO[ii][jj].setLevel(7);
        }
    }

    QKT(context, eval, pack, cipherA, cipherB, cipherO, m, n, num_inputs, num_QKT_head_cts, num_QKT_cts, head_dim, log_slots, num_slots, is_gpu);

    std::cout << "=== Plaintext QKT Result ===" << std::endl;
    for(int ii = 0; ii < m; ii++) {
        for(int jj = 0; jj < m; jj++) {
            double tmp = 0;
            int len = (n <= head_dim) ? n : head_dim;
            int iter = (n <= head_dim) ? int(head_dim / n) : 1;
            for(int kk = 0; kk < len; kk++) {
                for(int ll = 0; ll < iter; ll++) {
                    tmp += matrixA[ii][kk] * matrixB[jj][kk];
                }
            }
            matrixO[ii][jj] = tmp;
        }
    }

    for(int ii = 0; ii < out_size; ii++) {
        for(int jj = 0; jj < out_size; jj++) {
            std::cout << std::fixed << std::setprecision(2) << matrixO[ii][jj] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "=== Encrypted QKT Result ===" << std::endl;
    for(int kk = 0; kk < num_QKT_cts; kk++) {
        HEaaN::Message result_msg;
        dec.decrypt(cipherO[0][kk], sk, result_msg);
        int len = (n <= head_dim) ? n : m;
        for(int ii = 0; ii < m; ii++) {
            for(int jj = 0; jj < len; jj++) {
                matrixOC[ii][kk * len + jj] = result_msg[ii * n + jj].real();
            }
		}
    }

    for(int ii = 0; ii < out_size; ii++) {
        for(int jj = 0; jj < out_size; jj++) {
            std::cout << std::fixed << std::setprecision(2) << matrixOC[ii][jj] << " ";
        }
        std::cout << std::endl;
    }

    // Replacements of Softmax for testing QKTV
    for(int input = 0; input < num_QKT_head_cts; input++) {
        for(int ii = 0; ii < num_QKT_cts; ii++) {
            cipherO[input][ii].setLevel(5);
        }
    }

    std::cout << "=== QKTV computation (SRBMM) ===" << std::endl;

    std::vector<HEaaN::Ciphertext> cipherO2(num_inputs, HEaaN::Ciphertext(context));
    for(int ii = 0; ii < num_inputs; ii++) {
        enc.encrypt(zero, pack, cipherO2[ii]);
        cipherO2[ii].setLevel(3);
    }

    QKTV(context, eval, pack, cipherO, cipherC, cipherO2, m, n, num_inputs, num_QKT_head_cts, num_QKT_cts, head_dim, log_slots, num_slots, is_gpu);

    std::cout << "=== Plaintext QKTV Result ===" << std::endl;
    for(int ii = 0; ii < m; ii++) {
        int len = (n <= head_dim) ? n : head_dim;
        for(int jj = 0; jj < len; jj++) {
            double tmp = 0;
            for(int kk = 0; kk < m; kk++) {
                tmp += matrixO[ii][kk] * matrixC[kk][jj];
            }
            matrixO2[ii][jj] = tmp;
        }
    }

    for(int ii = 0; ii < out_size; ii++) {
        for(int jj = 0; jj < out_size; jj++) {
            std::cout << std::fixed << std::setprecision(2) << matrixO2[ii][jj] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "=== Encrypted QKTV Result ===" << std::endl;
    HEaaN::Message result_msg;
    dec.decrypt(cipherO2[0], sk, result_msg);
    for(int ii = 0; ii < m; ii++) {
        for(int jj = 0; jj < n; jj++) {
            matrixOC2[ii][jj] = result_msg[ii * n + jj].real();
        }
    }

    for(int ii = 0; ii < out_size; ii++) {
        for(int jj = 0; jj < out_size; jj++) {
            std::cout << std::fixed << std::setprecision(2) << matrixOC2[ii][jj] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
