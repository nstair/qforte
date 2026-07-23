#include <stdexcept>

#include "fmt/format.h"

#include "gate.h"

namespace {

bool local_bit(size_t idx, size_t qubit) {
    return (idx >> (3 - qubit)) & 1;
}

size_t set_local_bit(size_t idx, size_t qubit, bool value) {
    const size_t mask = static_cast<size_t>(1) << (3 - qubit);
    if (value) {
        return idx | mask;
    }
    return idx & ~mask;
}

void apply_local_givens(std::array<std::complex<double>, 16>& vec,
                        size_t source, size_t target,
                        std::complex<double> c, std::complex<double> s) {
    // Match the historical 2-qubit Givens convention: matrix basis is
    // [source, target], and |10> -> c|10> + s|01> at the vector level.
    for (size_t idx = 0; idx < 16; ++idx) {
        if (local_bit(idx, source) and not local_bit(idx, target)) {
            const size_t idx_source = idx;
            size_t idx_target = set_local_bit(idx, source, false);
            idx_target = set_local_bit(idx_target, target, true);

            const auto source_amp = vec[idx_source];
            const auto target_amp = vec[idx_target];
            vec[idx_source] = c * source_amp - s * target_amp;
            vec[idx_target] = s * source_amp + c * target_amp;
        }
    }
}

bool has_repeated_qubit(size_t q1, size_t q2, size_t q3, size_t q4) {
    return (q1 == q2) or (q1 == q3) or (q1 == q4) or
           (q2 == q3) or (q2 == q4) or (q3 == q4);
}

} // namespace

Gate make_gate(std::string type, size_t q1, size_t q2, size_t q3, size_t q4,
               std::complex<double> parameter) {
    if (has_repeated_qubit(q1, q2, q3, q4)) {
        std::string msg = fmt::format(
            "make_gate()\t{} requires four distinct qubits, got {}, {}, {}, {}",
            type, q1, q2, q3, q4);
        throw std::invalid_argument(msg);
    }

    if (type == "QNP_OR") {
        std::complex<double> c = std::cos(0.5 * parameter);
        std::complex<double> s = std::sin(0.5 * parameter);
        std::complex<double> gate[16][16]{};

        // QNP_OR is an orbital rotation on two qubit pairs.  In the local basis
        // [q1, q2, q3, q4], this is Givens(q1 -> q3) followed by Givens(q2 -> q4).
        for (size_t j = 0; j < 16; ++j) {
            std::array<std::complex<double>, 16> column{};
            column[j] = 1.0;
            apply_local_givens(column, 0, 2, c, s);
            apply_local_givens(column, 1, 3, c, s);
            for (size_t i = 0; i < 16; ++i) {
                gate[i][j] = column[i];
            }
        }

        return Gate(type, std::vector<size_t>{q1, q2, q3, q4}, gate);
    }
    if (type == "QNP_PX") {
        std::complex<double> c = std::cos(0.5 * parameter);
        std::complex<double> s = std::sin(0.5 * parameter);
        std::complex<double> gate[16][16]{};
        for (size_t i = 0; i < 16; ++i) {
            gate[i][i] = 1.0;
        }

        // In the local basis [q1, q2, q3, q4], QNP_PX is the diagonal-pair
        // Givens block between |0011> and |1100>. This is the matrix-level
        // effect of the controlled central Givens and its surrounding CNOTs.
        const size_t pair_01 = 3;  // |0011>
        const size_t pair_10 = 12; // |1100>
        gate[pair_01][pair_01] = c;
        gate[pair_01][pair_10] = s;
        gate[pair_10][pair_01] = -s;
        gate[pair_10][pair_10] = c;

        return Gate(type, std::vector<size_t>{q1, q2, q3, q4}, gate);
    }

    std::string msg =
        fmt::format("make_gate()\ntype = {} is not a valid 4-qubit quantum gate type", type);
    throw std::invalid_argument(msg);
}

Gate make_gate(std::string type, size_t target, size_t control, std::complex<double> parameter) {
    //using namespace std::complex_literals;
    std::complex<double> onei(0.0, 1.0);
    if (target == control) {
        if (type == "X") {
            std::complex<double> gate[4][4]{
                {0.0, 1.0},
                {1.0, 0.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "Y") {
            std::complex<double> gate[4][4]{
                {0.0, -onei},
                {+onei, 0.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "Z") {
            std::complex<double> gate[4][4]{
                {+1.0, 0.0},
                {0.0, -1.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "H") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {+c, +c},
                {+c, -c},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "R") {
	    std::complex<double> tmp = onei * parameter;
            std::complex<double> c = std::exp(tmp);
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, c},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "Rx") {
            std::complex<double> a = std::cos(0.5 * parameter);
            std::complex<double> b = onei * std::sin(0.5 * parameter);
            std::complex<double> gate[4][4]{
                {+a, -b},
                {-b, +a},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "Ry") {
            std::complex<double> a = std::cos(0.5 * parameter);
            std::complex<double> b = std::sin(0.5 * parameter);
            std::complex<double> gate[4][4]{
                {+a, -b},
                {+b, +a},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "Rz") {
            std::complex<double> tmp_a = -onei * 0.5 * parameter;
            std::complex<double> a = std::exp(tmp_a);
            std::complex<double> tmp_b = onei * 0.5 * parameter;
            std::complex<double> b = std::exp(tmp_b);
            std::complex<double> gate[4][4]{
                {a, 0.0},
                {0.0, b},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "V") {
            std::complex<double> a = onei * 0.5 + 0.5;
            std::complex<double> b = -onei * 0.5 + 0.5;
            std::complex<double> gate[4][4]{
                {+a, +b},
                {+b, +a},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "S") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, onei},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "T") {
            std::complex<double> c = (1.0 + onei) / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, c},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "I") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, 1.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "Rzy") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
            std::complex<double> c_i = onei / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {+c_i, +c},
                {+c, +c_i},
            };
            return Gate(type, target, control, gate);
        } if (type == "rU1") {
            std::complex<double> a = std::cos(parameter);
            std::complex<double> b = std::sin(parameter);
            std::complex<double> gate[4][4]{
                {+a, -b},
                {+b, +a},
            };
            return Gate(type, target, control, gate);
        }

    } else {
        if (type == "A") {
            // std::complex<double> c = std::cos(2.0*parameter);
            // std::complex<double> s = -onei*std::sin(2.0*parameter);
            std::complex<double> c = std::cos(parameter);
            std::complex<double> s = std::sin(parameter);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, c  ,  s,  0.0},
                {0.0, s  , -c,  0.0},
                {0.0, 0.0, 0.0, 1.0},
            };
            // std::complex<double> gate[4][4]{
            //     {1.0, 0.0, 0.0, 0.0},
            //     {0.0, c  ,  s,  0.0},
            //     {0.0, s  ,  c,  0.0},
            //     {0.0, 0.0, 0.0, 1.0},
            // };
            return Gate(type, target, control, gate);
        }
        if ((type == "Givens") or (type == "G")) {
            std::complex<double> c = std::cos(0.5 * parameter);
            std::complex<double> s = std::sin(0.5 * parameter);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, c,   s,  0.0},
                {0.0, -s,  c,  0.0},
                {0.0, 0.0, 0.0, 1.0},
            };
            return Gate(type, target, control, gate);
        }
        if ((type == "cX") or (type == "CNOT")) {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
                {0.0, 0.0, 1.0, 0.0},
            };
            return Gate(type, target, control, gate);
        }
        if ((type == "acX") or (type == "aCNOT")) {
            std::complex<double> gate[4][4]{
                {0.0, 1.0, 0.0, 0.0},
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "cY") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, -onei},
                {0.0, 0.0, +onei, 0.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "cZ") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, -1.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "cR") {
            std::complex<double> tmp = onei * parameter;
            std::complex<double> c = std::exp(tmp);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, c},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "cV") {
            std::complex<double> a = onei * 0.5 + 0.5;
            std::complex<double> b = -onei * 0.5 + 0.5;
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, +a, +b},
                {0.0, 0.0, +b, +a},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "cRz") {
            std::complex<double> tmp_a = -onei * 0.5 * parameter;
            std::complex<double> a = std::exp(tmp_a);
            std::complex<double> tmp_b = onei * 0.5 * parameter;
            std::complex<double> b = std::exp(tmp_b);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, a,   0.0},
                {0.0, 0.0, 0.0, b},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "SWAP") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
            };
            return Gate(type, target, control, gate);
        } if (type == "rU2") {
            std::complex<double> a = std::cos(parameter);
            std::complex<double> b = std::sin(parameter);
            std::complex<double> gate[4][4]{
                { +a,  -b, 0.0, 0.0},
                { +b,  +a, 0.0, 0.0},
                {0.0, 0.0,  +a,  -b},
                {0.0, 0.0,  +b,  +a},
            };
            return Gate(type, target, control, gate);
        }
    }
    // If you reach this section then the gate type is not implemented or it is invalid.
    // So we throw an exception that propagates to Python and return the identity
    std::string msg =
        fmt::format("make_gate()\ntype = {} is not a valid quantum gate type", type);
    throw std::invalid_argument(msg);
    std::complex<double> gate[4][4]{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
    };
    return Gate(type, target, control, gate);
}

Gate make_control_gate(size_t control, Gate& U) {
    //using namespace std::complex_literals;
    std::string type = "cU";
    size_t target = U.target();
    if (target == control) {
        std::string msg =
            fmt::format("Cannot create Control-U where targer == control !");
        throw std::invalid_argument(msg);
    }
    std::complex<double> a = U.gate()[0][0];
    std::complex<double> b = U.gate()[0][1];
    std::complex<double> c = U.gate()[1][0];
    std::complex<double> d = U.gate()[1][1];
    std::complex<double> gate[4][4]{
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, a, b},
            {0.0, 0.0, c, d},
        };
    return Gate(type, target, control, gate);
}
