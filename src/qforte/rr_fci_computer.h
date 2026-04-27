// rr_fci_computer.h
#ifndef _rr_fci_computer_h_
#define _rr_fci_computer_h_

#include <string>
#include <vector>
#include <complex>
#include <stdexcept>
#include <utility>

#include "qforte-def.h"
#include "tensor.h"
#include "fci_graph.h"
#include "timer.h"
#include "df_hamiltonian.h"

class local_timer;
class Tensor;
class FCIGraph;

class RRFCIComputer {
  public:
    /// Reduced-rank FCI computer.
    ///
    /// Physical storage is row-major with shapes
    ///   P_ : {R_, nalfa_strs_}
    ///   Q_ : {R_, nbeta_strs_}
    ///
    /// so that for fixed rank r, the full string slice is contiguous.
    RRFCIComputer(int nel, int sz, int norb, int rank);

    // ----- element access -----
    //
    // idxs = {which, string_index, rank_index}
    // which = 0 => P(string_index, rank_index)
    // which = 1 => Q(string_index, rank_index)
    //
    // Logically this is P_J^r / Q_K^r, even though physical storage is transposed.
    void set_element(
        const std::vector<size_t>& idxs,
        const std::complex<double> val);

    void add_to_element(
        const std::vector<size_t>& idxs,
        const std::complex<double> val);

    void set_p_element(size_t J, size_t r, const std::complex<double> val);
    void set_q_element(size_t K, size_t r, const std::complex<double> val);

    std::complex<double> get_p_element(size_t J, size_t r) const;
    std::complex<double> get_q_element(size_t K, size_t r) const;

    // ----- Hamiltonian application entry points -----
    void apply_tensor_spat_12bdy(
        const Tensor& h1e,
        const Tensor& h2e,
        const Tensor& h2e_einsum,
        size_t norb);

    void apply_tensor_spat_012bdy(
        const std::complex<double> h0e,
        const Tensor& h1e,
        const Tensor& h2e,
        const Tensor& h2e_einsum,
        size_t norb);

    // ----- More projected sigma helpers -----
    void rr_apply_same_spin_alpha_to_sigmaP(
        Tensor& outP,
        const Tensor& leftQ,
        const Tensor& rightQ,
        const Tensor& rightP,
        const Tensor& h1e,
        const Tensor& h2e,
        const int norbs);

    void rr_apply_same_spin_beta_to_sigmaP(
        Tensor& outP,
        const Tensor& leftQ,
        const Tensor& rightQ,
        const Tensor& rightP,
        const Tensor& h1e,
        const Tensor& h2e,
        const int norbs);

    void rr_apply_same_spin_alpha_to_sigmaQ(
        Tensor& outQ,
        const Tensor& leftP,
        const Tensor& rightP,
        const Tensor& rightQ,
        const Tensor& h1e,
        const Tensor& h2e,
        const int norbs);

    void rr_apply_same_spin_beta_to_sigmaQ(
        Tensor& outQ,
        const Tensor& leftP,
        const Tensor& rightP,
        const Tensor& rightQ,
        const Tensor& h1e,
        const Tensor& h2e,
        const int norbs);

    // ----- projected sigma helpers -----
    void rr_apply_array12_same_spin_opt_P(
        Tensor& outP,
        const Tensor& leftQ,
        const Tensor& rightQ,
        const Tensor& rightP,
        const std::vector<int>& dexc,
        const int alpha_states,
        const int ndexc,
        const Tensor& h1e,
        const Tensor& h2e,
        const int norbs);

    void rr_apply_array12_same_spin_opt_Q(
        Tensor& outQ,
        const Tensor& leftP,
        const Tensor& rightP,
        const Tensor& rightQ,
        const std::vector<int>& dexc,
        const int beta_states,
        const int ndexc,
        const Tensor& h1e,
        const Tensor& h2e,
        const int norbs);

    void rr_apply_array12_diff_spin_opt_P(
        Tensor& outP,
        const Tensor& leftQ,
        const Tensor& rightQ,
        const Tensor& rightP,
        const std::vector<int>& adexc,
        const std::vector<int>& bdexc,
        const int alpha_states,
        const int beta_states,
        const int nadexc,
        const int nbdexc,
        const Tensor& h2e_einsum,
        const int norbs);

    void rr_apply_array12_diff_spin_opt_Q(
        Tensor& outQ,
        const Tensor& leftP,
        const Tensor& rightP,
        const Tensor& rightQ,
        const std::vector<int>& adexc,
        const std::vector<int>& bdexc,
        const int alpha_states,
        const int beta_states,
        const int nadexc,
        const int nbdexc,
        const Tensor& h2e_einsum,
        const int norbs);

    // ----- lower-level helpers in the call tree -----
    void apply_array_1bdy_vector(
        std::vector<std::complex<double>>& out,
        const std::vector<int>& dexc,
        const int states,
        const int ndexc,
        const Tensor& h1e,
        const std::vector<std::complex<double>>& in,
        const int norbs) const;

    void lm_apply_array12_same_spin_opt_vector(
        std::vector<std::complex<double>>& out,
        const std::vector<int>& dexc,
        const int states,
        const int ndexc,
        const Tensor& h1e,
        const Tensor& h2e,
        const std::vector<std::complex<double>>& in,
        const int norbs) const;

    void build_transition_vector(
        std::vector<std::complex<double>>& gamma_proj,
        const std::vector<int>& dexc,
        const int states,
        const int ndexc,
        const std::vector<std::complex<double>>& left,
        const std::vector<std::complex<double>>& right,
        const int norbs) const;

    void build_transition_vector_no_conj(
        std::vector<std::complex<double>>& gamma_proj,
        const std::vector<int>& dexc,
        const int states,
        const int ndexc,
        const std::vector<std::complex<double>>& left,
        const std::vector<std::complex<double>>& right,
        const int norbs) const;

    void build_effective_one_body_from_transition(
        Tensor& heff,
        const std::vector<std::complex<double>>& transition,
        const Tensor& h2e_einsum,
        const int norbs) const;

    void build_overlap_projections(Tensor& SP, Tensor& SQ) const;

    // ----- slice helpers -----
    std::vector<std::complex<double>> get_P_row(size_t r) const;
    std::vector<std::complex<double>> get_Q_row(size_t r) const;

    void set_P_row(size_t r, const std::vector<std::complex<double>>& row);
    void set_Q_row(size_t r, const std::vector<std::complex<double>>& row);

    void axpy_into_row(
        Tensor& T,
        size_t row,
        const std::vector<std::complex<double>>& vec,
        const std::complex<double>& alpha);

    std::complex<double> dot_vec(
        const std::vector<std::complex<double>>& a,
        const std::vector<std::complex<double>>& b) const;

    std::complex<double> contract_vec_no_conj(
        const std::vector<std::complex<double>>& a,
        const std::vector<std::complex<double>>& b) const;

    // ----- getters -----
    size_t get_nel() const { return nel_; }
    size_t get_nalfa_el() const { return nalfa_el_; }
    size_t get_nbeta_el() const { return nbeta_el_; }
    size_t get_nalfa_strs() const { return nalfa_strs_; }
    size_t get_nbeta_strs() const { return nbeta_strs_; }
    size_t get_sz() const { return sz_; }
    size_t get_norb() const { return norb_; }
    size_t get_rank() const { return R_; }

    const Tensor& P() const { return P_; }
    const Tensor& Q() const { return Q_; }
    const Tensor& SigmaP() const { return SigmaP_; }
    const Tensor& SigmaQ() const { return SigmaQ_; }

    Tensor& P() { return P_; }
    Tensor& Q() { return Q_; }
    Tensor& SigmaP() { return SigmaP_; }
    Tensor& SigmaQ() { return SigmaQ_; }

    FCIGraph get_graph() const { return graph_; }
    std::vector<std::pair<std::string, double>> get_timings() const { return timings_; }
    local_timer get_acc_timer() const { return timer_; }

    // ----- setters -----
    void set_rank(size_t rank);
    void set_P(const Tensor& P);
    void set_Q(const Tensor& Q);

    void zero();
    void clear_sigmas();
    void clear_timings() { timings_.clear(); }

    // void do_on_gpu();
    // void do_on_cpu();

    void hartree_fock();

    std::complex<double> get_hf_dot() const;

    std::complex<double> get_overlap(const RRFCIComputer& other) const;

    std::complex<double> pq_dot_product(const Tensor& Pother, const Tensor& Qother) const;

    std::complex<double> get_exp_val_tensor(
        const std::complex<double> h0e,
        const Tensor& h1e,
        const Tensor& h2e,
        const Tensor& h2e_einsum,
        size_t norb);

    Tensor reconstruct_C() const;

  private:
    // ----- flat index helpers -----
    inline size_t p_index(size_t r, size_t J) const { return r * nalfa_strs_ + J; }
    inline size_t q_index(size_t r, size_t K) const { return r * nbeta_strs_ + K; }

    bool use_gpu_operations_ = false;

    size_t nel_;
    size_t nalfa_el_;
    size_t nbeta_el_;
    size_t nalfa_strs_;
    size_t nbeta_strs_;
    size_t sz_;
    size_t norb_;
    size_t nabasis_ = 0;
    size_t nbbasis_ = 0;

    size_t R_;

    const std::string name_ = "RRFCIComputer State";

    /// Physical layout:
    ///   P_(r, J) stored row-major as shape {R_, nalfa_strs_}
    ///   Q_(r, K) stored row-major as shape {R_, nbeta_strs_}
    Tensor P_;
    Tensor Q_;

    /// Persistent work buffers with same physical layout
    Tensor SigmaP_;
    Tensor SigmaQ_;

    FCIGraph graph_;
    local_timer timer_;
    std::vector<std::pair<std::string, double>> timings_;

    double compute_threshold_ = 1.0e-12;
};

#endif // _rr_fci_computer_h_