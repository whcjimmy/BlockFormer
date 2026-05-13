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
    // HEaaN::makeBootstrappable(context);    // GPU MODE
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

    std::string str = argv[1]; // name of the non-linear function
    int m = atoi(argv[2]);     // input length
    int alpha = atoi(argv[3]); // BP parameter
    int head_dim = 64; 
    int emb_dim = 768; 
    int hidden_dim = 3072;
    int is_gpu = atoi(argv[4]) == 1 ? true : false;

    int num_heads = emb_dim / head_dim; // 768/64=12
    int num_inputs;
    if(str == "softmax"){
        num_inputs = num_heads * m / alpha;
    } else if(str == "ln_10" || str == "ln_4" || str == "bs") {
        num_inputs = num_heads * head_dim / alpha;
    }else if (str == "gelu") {
        num_inputs = hidden_dim / alpha;
    }

    std::cout << "=========\n";
    if(is_gpu) {
        std::cout << "GPU MODE\n";
    }else {
        std::cout << "CPU MODE\n";
    }
    std::cout << "=========\n";

    std::cout << "NUM INPUTS: " << num_inputs << std::endl;

    std::vector<HEaaN::Ciphertext> vec_cipher(num_inputs, HEaaN::Ciphertext(context));

    HEaaN::Message zero(log_slots);
    fillZero(zero);

    // some random numbers
    std::vector<HEaaN::Message> msg(5, HEaaN::Message(log_slots));
    for(int i = 0; i < 5; i++) {
        fillRandomComplex(msg[i]);
    }

    int enc_level;
    if(str == "bs") {
        enc_level = 3;
    }else if( str == "softmax") {
        enc_level = 7;
    }else if( str == "gelu") {
        enc_level = 6;
    }else if( str == "ln_10") {
        enc_level = 10;
    }else if( str == "ln_4") {
        enc_level = 4;
    }

#pragma omp parallel for
    for(int i = 0; i < num_inputs; i++) {
        HEaaN::Message randnum(log_slots);
        for(int j = 0; j < log_slots; j++) {
            randnum[j].real((double)rand() / RAND_MAX);
            randnum[j].imag(0.0);
        }
        enc.encrypt(randnum, pack, vec_cipher[i]);
        vec_cipher[i].setLevel(enc_level);
    }

    if(is_gpu) {
        for(int i = 0; i < num_inputs; i++) {
            vec_cipher[i].to(HEaaN::DeviceType::GPU);
        }
        for(int i = 0 ; i < msg.size(); i++) {
            msg[i].to(HEaaN::DeviceType::GPU);
        }
    }

    if(str == "bs") {
        std::cout << "=== Bootstrapping ===" << std::endl;
        timer.start("bootstrapping");
#pragma omp parallel for if(!is_gpu)
        for(int i = 0; i < num_inputs; i++) {
            boot.bootstrap(vec_cipher[i], vec_cipher[i]);          // CPU MODE
            // eval.bootstrap(vec_cipher[i], vec_cipher[i], true); // GPU MODE
        }
        timer.end();
        timer.print();
    }else if (str == "softmax") {
        std::cout << "=== Softmax ===" << std::endl;
        timer.start("softmax");
#pragma omp parallel for if(!is_gpu)
        for(int i = 0; i < num_inputs; i++) {
            HEaaN::Ciphertext x1_cipher(context);
            eval.add(vec_cipher[i], msg[0], x1_cipher); // x + c
            eval.mult(x1_cipher, x1_cipher, x1_cipher); // (x + c)^2 
            eval.mult(x1_cipher, x1_cipher, x1_cipher); // (x + c)^4
            vec_cipher[i] = x1_cipher;
        }
        timer.end();
        timer.print();
    }else if (str == "gelu") {
        std::cout << "=== GeLU ===" << std::endl;
        timer.start("gelu");
#pragma omp parallel for if(!is_gpu)
        for(int i = 0; i < num_inputs; i++) {
            HEaaN::Ciphertext tmp_cipher(context);
            eval.mult(vec_cipher[i], vec_cipher[i], tmp_cipher); // x^2
            eval.mult(vec_cipher[i], msg[1], vec_cipher[i]);     // bx
            eval.add(vec_cipher[i], tmp_cipher, vec_cipher[i]);  // x^2 + bx
            eval.add(vec_cipher[i], msg[2], vec_cipher[i]);      // x^2 + bx + c
        }
        timer.end();
        timer.print();
    }else if (str == "ln_10" || str == "ln_4") {
        int batch_size = 32768 / (m * alpha);
        std::cout << "Batch size for LN: " << batch_size << std::endl;

        HEaaN::Ciphertext sum_cipher(context), tmp_cipher(context); 
        enc.encrypt(zero, pack, sum_cipher);
        sum_cipher.setLevel(enc_level);

        HEaaN::Message filter_msg(log_slots);
        fillZero(filter_msg);
        int t = 0;
        while(t < num_slots) {
            filter_msg[t].real(1.0);
            filter_msg[t].imag(0.0);
            t+= alpha;
        }

        if(is_gpu) {
            sum_cipher.to(HEaaN::DeviceType::GPU);
            tmp_cipher.to(HEaaN::DeviceType::GPU);
            filter_msg.to(HEaaN::DeviceType::GPU);
        }

        std::cout << "=== LN ===" << std::endl;
        timer.start("ln");

        // sum 768 elements of each row
        for(int i = 0; i < num_inputs; i++) {
            eval.add(sum_cipher, vec_cipher[i], sum_cipher);
        }

        tmp_cipher = sum_cipher;
        for(int ii = log2(alpha) - 1; ii >=0; ii--) {
            eval.leftRotate(sum_cipher, pow(2, ii), sum_cipher);
            eval.add(tmp_cipher, sum_cipher, tmp_cipher);
            sum_cipher = tmp_cipher;
        }

        // only keep the sum of each row 
        eval.mult(sum_cipher, filter_msg, sum_cipher); // ax
        
        // broadcast the sum to all slots
        tmp_cipher = sum_cipher;
        for(int ii = 0; ii < log2(alpha); ii++) {
            eval.rightRotate(sum_cipher, pow(2, ii), sum_cipher);
            eval.add(tmp_cipher, sum_cipher, tmp_cipher);
            sum_cipher = tmp_cipher;
        }

#pragma omp parallel for if(!is_gpu)
        for(int i = 0; i < num_inputs; i++) {
            vec_cipher[i].setLevel(enc_level - 1);
            eval.sub(vec_cipher[i], sum_cipher, vec_cipher[i]); // x - u
            eval.add(vec_cipher[i], msg[0], vec_cipher[i]); // (x - u) + b
        }
        timer.end();
        timer.print();

        if(is_gpu) {
            sum_cipher.to(HEaaN::DeviceType::CPU);
            tmp_cipher.to(HEaaN::DeviceType::CPU);
            filter_msg.to(HEaaN::DeviceType::CPU);
        }
    }

    if(is_gpu) {
        for(int i = 0; i < num_inputs; i++) {
            vec_cipher[i].to(HEaaN::DeviceType::CPU);
        }
        for(int i = 0 ; i < msg.size(); i++) {
            msg[i].to(HEaaN::DeviceType::CPU);
        }
    }

    return 0;
}
