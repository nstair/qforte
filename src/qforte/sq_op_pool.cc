#include "helpers.h"
#include "gate.h"
#include "circuit.h"
#include "qubit_operator.h"
#include "sq_operator.h"
#include "qubit_op_pool.h"
#include "sq_op_pool.h"
#include "df_hamiltonian.h"
#include "tensor.h"

#include "qubit_basis.h"

#include <stdexcept>
#include <algorithm>

void SQOpPool::add_term(std::complex<double> coeff, const SQOperator& sq_op ){
    terms_.push_back(std::make_pair(coeff, sq_op));
}

/// NICK: This funcion is working but needs testing for edge cases!!
void SQOpPool::add_hermitian_pairs(std::complex<double> coeff, const SQOperator& sq_op ){
    std::vector<std::pair< std::vector<size_t>, std::vector<size_t>>> h_vec;
    std::vector<std::pair< std::vector<size_t>, std::vector<size_t>>> hd_vec;

    for (size_t l = 0; l < sq_op.terms().size(); l++){
        std::pair< std::vector<size_t>, std::vector<size_t>> h;
        std::pair< std::vector<size_t>, std::vector<size_t>> hd;

        std::complex<double> hl = std::get<0>(sq_op.terms()[l]);
        h.first  = std::get<1>(sq_op.terms()[l]);
        h.second = std::get<2>(sq_op.terms()[l]);

        hd.first = h.second;
        hd.second = h.first;

        std::reverse(hd.first.begin(), hd.first.end());
        std::reverse(hd.second.begin(), hd.second.end());

        std::sort(h.first.begin(), h.first.end());
        std::sort(h.second.begin(), h.second.end());

        std::sort(hd.first.begin(), hd.first.end());
        std::sort(hd.second.begin(), hd.second.end());
        
        // Determine if term is in current set of terms or term adjoints
        // if it isn't found in either then append the vectors
        if (std::find(h_vec.begin(), h_vec.end(), h) == h_vec.end()){
            if (std::find(hd_vec.begin(), hd_vec.end(), h) == hd_vec.end()){
                SQOperator temp;
                // if term is same as term adjoint add both
                if(h == hd or h.first == h.second){
                    // (Nick) Need this checked out for sure
                    temp.add_term(hl/2.0, h.first, h.second);
                    temp.add_term(hl/2.0, hd.first, hd.second);
                } else {
                    temp.add_term(hl, h.first, h.second);
                    temp.add_term(hl, hd.first, hd.second);
                    temp.simplify();
                }

                terms_.push_back(std::make_pair(coeff, temp));

                h_vec.push_back(h);
                hd_vec.push_back(hd);

            }
        } 
    }
}

void SQOpPool::set_coeffs(const std::vector<std::complex<double>>& new_coeffs){
    if(new_coeffs.size() != terms_.size()){
        throw std::invalid_argument( "Number of new coefficients for quantum operator must equal." );
    }
    for (size_t l = 0; l < new_coeffs.size(); l++){
        terms_[l].first = new_coeffs[l];
    }
}

void SQOpPool::set_coeffs_to_scaler(std::complex<double> new_coeff){
    for (size_t l = 0; l < terms_.size(); l++){
        terms_[l].first = new_coeff;
    }
}

const std::vector<std::pair< std::complex<double>, SQOperator>>& SQOpPool::terms() const{
    return terms_;
}

void SQOpPool::set_orb_spaces(const std::vector<int>& ref){
    int norb = ref.size();
    if (norb%2 == 0){
        norb = static_cast<int>(norb/2);
    } else {
        throw std::invalid_argument("QForte does not yet support systems with an odd number of spin orbitals.");
    }

    nocc_ = 0;
    for (const auto& occupancy : ref){
        nocc_ += occupancy;
    }

    if (nocc_%2 == 0){
        nocc_ = static_cast<int>(nocc_/2);
    } else {
        throw std::invalid_argument("QForte does not yet support systems with an odd number of occupied spin orbitals.");
    }

    nvir_ = static_cast<int>(norb - nocc_);
}

QubitOpPool SQOpPool::get_qubit_op_pool(){
    QubitOpPool A;
    for (auto& term : terms_) {
        // QubitOperator a = term.second.jw_transform();
        // a.mult_coeffs(term.first);
        A.add_term(term.first, term.second.jw_transform());
    }
    return A;
}


QubitOperator SQOpPool::get_qubit_operator(const std::string& order_type, bool combine_like_terms, bool qubit_excitations){
    QubitOperator parent;

    if(order_type=="unique_lex"){
        for (auto& term : terms_) {
            auto child = term.second.jw_transform(qubit_excitations);
            child.mult_coeffs(term.first);
            parent.add_op(child);
        }
        // TODO: analyze ordering here, eliminating simplify will place commuting
        // terms closer together but may introduce redundancy.
        parent.simplify();
        parent.order_terms();
    } else if (order_type=="commuting_grp_lex") {
        for (auto& term : terms_) {
            auto child = term.second.jw_transform(qubit_excitations);
            child.mult_coeffs(term.first);
            child.simplify(combine_like_terms=combine_like_terms);
            child.order_terms();
            parent.add_op(child);

        }
    } else {
        throw std::invalid_argument( "Invalid order_type specified.");
    }
    return parent;
}

void SQOpPool::fill_pool(std::string pool_type){
    if(pool_type=="GSD"){
        size_t norb = nocc_ + nvir_;
        for(size_t i=0; i<norb; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;
            for (size_t a=i; a<norb; a++){
                size_t aa = 2*a;
                size_t ab = 2*a+1;

                if( aa != ia ){
                    SQOperator temp1a;
                    temp1a.add_term(+1.0, {aa}, {ia});
                    temp1a.add_term(-1.0, {ia}, {aa});
                    temp1a.simplify();
                    if(temp1a.terms().size() > 0){
                        add_term(1.0, temp1a);
                    }
                }

                if( ab != ib ){
                    SQOperator temp1b;
                    temp1b.add_term(+1.0, {ab}, {ib});
                    temp1b.add_term(-1.0, {ib}, {ab});
                    temp1b.simplify();
                    if(temp1b.terms().size() > 0){
                        add_term(1.0, temp1b);
                    }
                }
            }
        }

        std::vector< std::vector<size_t> > uniqe_2bdy;
        std::vector< std::vector<size_t> > adjnt_2bdy;

        for(size_t i=0; i<norb; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;
            for(size_t j=i; j<norb; j++){
                size_t ja = 2*j;
                size_t jb = 2*j+1;
                for(size_t a=0; a<norb; a++){
                    size_t aa = 2*a;
                    size_t ab = 2*a+1;
                    for(size_t b=a; b<norb; b++){
                        size_t ba = 2*b;
                        size_t bb = 2*b+1;

                        if((aa != ba) && (ia != ja)){
                            SQOperator temp2aaaa;
                            temp2aaaa.add_term(+1.0, {aa,ba}, {ia,ja});
                            temp2aaaa.add_term(-1.0, {ja,ia}, {ba,aa});
                            temp2aaaa.simplify();
                            if(temp2aaaa.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2aaaa.terms()[0])[0], std::get<1>(temp2aaaa.terms()[0])[1], std::get<2>(temp2aaaa.terms()[0])[0], std::get<2>(temp2aaaa.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2aaaa.terms()[1])[0], std::get<1>(temp2aaaa.terms()[1])[1], std::get<2>(temp2aaaa.terms()[1])[0], std::get<2>(temp2aaaa.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_term(1.0, temp2aaaa);
                                    }
                                }
                            }
                        }

                        if((ab != bb ) && (ib != jb)){
                            SQOperator temp2bbbb;
                            temp2bbbb.add_term(+1.0, {ab,bb}, {ib,jb});
                            temp2bbbb.add_term(-1.0, {jb,ib}, {bb,ab});
                            temp2bbbb.simplify();
                            if(temp2bbbb.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2bbbb.terms()[0])[0], std::get<1>(temp2bbbb.terms()[0])[1], std::get<2>(temp2bbbb.terms()[0])[0], std::get<2>(temp2bbbb.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2bbbb.terms()[1])[0], std::get<1>(temp2bbbb.terms()[1])[1], std::get<2>(temp2bbbb.terms()[1])[0], std::get<2>(temp2bbbb.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_term(1.0, temp2bbbb);
                                    }
                                }
                            }
                        }

                        if((aa != bb) && (ia != jb)){
                            SQOperator temp2abab;
                            temp2abab.add_term(+1.0, {aa,bb}, {ia,jb});
                            temp2abab.add_term(-1.0, {jb,ia}, {bb,aa});
                            temp2abab.simplify();
                            if(temp2abab.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2abab.terms()[0])[0], std::get<1>(temp2abab.terms()[0])[1], std::get<2>(temp2abab.terms()[0])[0], std::get<2>(temp2abab.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2abab.terms()[1])[0], std::get<1>(temp2abab.terms()[1])[1], std::get<2>(temp2abab.terms()[1])[0], std::get<2>(temp2abab.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_term(1.0, temp2abab);
                                    }
                                }
                            }
                        }

                        if((ab != ba) && (ib != ja)){
                            SQOperator temp2baba;
                            temp2baba.add_term(+1.0, {ab,ba}, {ib,ja});
                            temp2baba.add_term(-1.0, {ja,ib}, {ba,ab});
                            temp2baba.simplify();
                            if(temp2baba.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2baba.terms()[0])[0], std::get<1>(temp2baba.terms()[0])[1], std::get<2>(temp2baba.terms()[0])[0], std::get<2>(temp2baba.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2baba.terms()[1])[0], std::get<1>(temp2baba.terms()[1])[1], std::get<2>(temp2baba.terms()[1])[0], std::get<2>(temp2baba.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_term(1.0, temp2baba);
                                    }
                                }
                            }
                        }

                        if((aa != bb) && (ib != ja)){
                            SQOperator temp2abba;
                            temp2abba.add_term(+1.0, {aa,bb}, {ib,ja});
                            temp2abba.add_term(-1.0, {ja,ib}, {bb,aa});
                            temp2abba.simplify();
                            if(temp2abba.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2abba.terms()[0])[0], std::get<1>(temp2abba.terms()[0])[1], std::get<2>(temp2abba.terms()[0])[0], std::get<2>(temp2abba.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2abba.terms()[1])[0], std::get<1>(temp2abba.terms()[1])[1], std::get<2>(temp2abba.terms()[1])[0], std::get<2>(temp2abba.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_term(1.0, temp2abba);
                                    }
                                }

                            }
                        }

                        if((ab != ba) && (ia != jb)){
                            SQOperator temp2baab;
                            temp2baab.add_term(+1.0, {ab,ba}, {ia,jb});
                            temp2baab.add_term(-1.0, {jb,ia}, {ba,ab});
                            temp2baab.simplify();
                            if(temp2baab.terms().size() > 0){
                                std::vector<size_t> vtemp {std::get<1>(temp2baab.terms()[0])[0], std::get<1>(temp2baab.terms()[0])[1], std::get<2>(temp2baab.terms()[0])[0], std::get<2>(temp2baab.terms()[0])[1]};
                                std::vector<size_t> vadjt {std::get<1>(temp2baab.terms()[1])[0], std::get<1>(temp2baab.terms()[1])[1], std::get<2>(temp2baab.terms()[1])[0], std::get<2>(temp2baab.terms()[1])[1]};
                                if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                                    if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                        uniqe_2bdy.push_back(vtemp);
                                        adjnt_2bdy.push_back(vadjt);
                                        add_term(1.0, temp2baab);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if ( (pool_type=="S") || (pool_type=="SD") || (pool_type=="SDT") || (pool_type=="SDTQ") || (pool_type=="SDTQP") || (pool_type=="SDTQPH") || (pool_type=="All") ) {

        int max_nbody = 0;

        if(pool_type=="S") {
            max_nbody = 1;
        } else if(pool_type=="SD") {
            max_nbody = 2;
        }else if(pool_type=="SDT") {
            max_nbody = 3;
        } else if(pool_type=="SDTQ") {
            max_nbody = 4;
        } else if(pool_type=="SDTQP") {
            max_nbody = 5;
        } else if(pool_type=="SDTQPH") {
            max_nbody = 6;
        } else if(pool_type=="All") {
            max_nbody = 2 * nocc_;
        } else {
            throw std::invalid_argument( "Qforte UCC only supports up to Hextuple excitations." );
        }

        int nqb = 2 * (nocc_ + nvir_);
        int nel = 2 * nocc_;

        // TODO(Nick): incorporate more flexibility into this
        int na_el = nocc_;
        int nb_el = nocc_;

        for (int I=0; I<std::pow(2, nqb); I++) {

            // get the basis state (I) | 11001100 > or whatever..
            QubitBasis basis_I(I);

            if(basis_I.get_num_ones() != na_el + nb_el) {
                continue;
            }

            int nbody = 0;
            int pn = 0;
            int na_I = 0;
            int nb_I = 0;
            std::vector<size_t> holes; // i, j, k, ...
            std::vector<size_t> particles; // a, b, c, ...
            std::vector<int> parity;

            for (size_t p=0; p<2*nocc_; p++) {
                int bit_val = static_cast<int>(basis_I.get_bit(p));
                nbody += ( 1 - bit_val);
                pn += bit_val;
                if(p%2==0){
                    na_I += bit_val;
                } else {
                    nb_I += bit_val;
                }

                if(bit_val-1){
                    holes.push_back(p);
                    if(p%2==0){
                        parity.push_back(1);
                    } else {
                        parity.push_back(-1);
                    }
                }
            }
            for (size_t q=2*nocc_; q<nqb; q++) {
                int bit_val = static_cast<int>(basis_I.get_bit(q));
                pn += bit_val;
                if(q%2==0){
                    na_I += bit_val;
                } else {
                    nb_I += bit_val;
                }
                if(bit_val){
                    particles.push_back(q);
                    if(q%2==0){
                        parity.push_back(1);
                    } else {
                        parity.push_back(-1);
                    }
                }
            }

            if(pn==nel && na_I == na_el && nb_I == nb_el){

                if (nbody != 0 && nbody <= max_nbody ) {

                    int total_parity = 1;
                    for (const auto& z: parity){
                        total_parity *= z;
                    }

                    if(total_parity==1){
                        // need i, j, a, b
                        SQOperator t_temp;
                        t_temp.add_term(+1.0, particles, holes);
                        std::vector<size_t> rparticles(particles.rbegin(), particles.rend());
                        std::vector<size_t> rholes(holes.rbegin(), holes.rend());
                        t_temp.add_term(-1.0, rholes, rparticles);
                        t_temp.simplify();
                        add_term(1.0, t_temp);
                    }
                }
            }
        }
    } else if(pool_type=="sa_SD"){
        for(size_t i=0; i<nocc_; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;

            for (size_t a=0; a<nvir_; a++){
                size_t aa = 2*nocc_ + 2*a;
                size_t ab = 2*nocc_ + 2*a+1;

                SQOperator temp1;
                temp1.add_term(+1.0/std::sqrt(2), {aa}, {ia});
                temp1.add_term(+1.0/std::sqrt(2), {ab}, {ib});

                temp1.add_term(-1.0/std::sqrt(2), {ia}, {aa});
                temp1.add_term(-1.0/std::sqrt(2), {ib}, {ab});

                temp1.simplify();

                add_term(1.0, temp1);
            }
        }

        for(size_t i=0; i<nocc_; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;

            for(size_t j=i; j<nocc_; j++){
                size_t ja = 2*j;
                size_t jb = 2*j+1;

                for(size_t a=0; a<nvir_; a++){
                    size_t aa = 2*nocc_ + 2*a;
                    size_t ab = 2*nocc_ + 2*a+1;

                    for(size_t b=a; b<nvir_; b++){
                        size_t ba = 2*nocc_ + 2*b;
                        size_t bb = 2*nocc_ + 2*b+1;

                        SQOperator temp2a;
                        if((aa != ba) && (ia != ja)){
                            temp2a.add_term(2.0/std::sqrt(12), {aa,ba}, {ia,ja});
                        }
                        if((ab != bb ) && (ib != jb)){
                            temp2a.add_term(2.0/std::sqrt(12), {ab,bb}, {ib,jb});
                        }
                        if((aa != bb) && (ia != jb)){
                            temp2a.add_term(1.0/std::sqrt(12), {aa,bb}, {ia,jb});
                        }
                        if((ab != ba) && (ib != ja)){
                            temp2a.add_term(1.0/std::sqrt(12), {ab,ba}, {ib,ja});
                        }
                        if((aa != bb) && (ib != ja)){
                            temp2a.add_term(1.0/std::sqrt(12), {aa,bb}, {ib,ja});
                        }
                        if((ab != ba) && (ia != jb)){
                            temp2a.add_term(1.0/std::sqrt(12), {ab,ba}, {ia,jb});
                        }

                        // hermitian conjugate
                        if((ja != ia) && (ba != aa)){
                            temp2a.add_term(-2.0/std::sqrt(12), {ja,ia}, {ba,aa});
                        }
                        if((jb != ib ) && (bb != ab)){
                            temp2a.add_term(-2.0/std::sqrt(12), {jb,ib}, {bb,ab});
                        }
                        if((jb != ia) && (bb != aa)){
                            temp2a.add_term(-1.0/std::sqrt(12), {jb,ia}, {bb,aa});
                        }
                        if((ja != ib) && (ba != ab)){
                            temp2a.add_term(-1.0/std::sqrt(12), {ja,ib}, {ba,ab});
                        }
                        if((ja != ib) && (bb != aa)){
                            temp2a.add_term(-1.0/std::sqrt(12), {ja,ib}, {bb,aa});
                        }
                        if((jb != ia) && (ba != ab)){
                            temp2a.add_term(-1.0/std::sqrt(12), {jb,ia}, {ba,ab});
                        }

                        SQOperator temp2b;
                        if((aa != bb) && (ia != jb)){
                            temp2b.add_term(0.5, {aa,bb}, {ia,jb});
                        }
                        if((ab != ba) && (ib != ja)){
                            temp2b.add_term(0.5, {ab,ba}, {ib,ja});
                        }
                        if((aa != bb) && (ib != ja)){
                            temp2b.add_term(-0.5, {aa,bb}, {ib,ja});
                        }
                        if((ab != ba) && (ia != jb)){
                            temp2b.add_term(-0.5, {ab,ba}, {ia,jb});
                        }

                        // hermetian conjugate
                        if((jb != ia) && (bb != aa)){
                            temp2b.add_term(-0.5, {jb,ia}, {bb,aa});
                        }
                        if((ja != ib) && (ba != ab)){
                            temp2b.add_term(-0.5, {ja,ib}, {ba,ab});
                        }
                        if((ja != ib) && (bb != aa)){
                            temp2b.add_term(0.5, {ja,ib}, {bb,aa});
                        }
                        if((jb != ia) && (ba != ab)){
                            temp2b.add_term(0.5, {jb,ia}, {ba,ab});
                        }

                        temp2a.simplify();
                        temp2b.simplify();

                        std::complex<double> temp2a_norm(0.0, 0.0);
                        std::complex<double> temp2b_norm(0.0, 0.0);
                        for (const auto& term : temp2a.terms()){
                            temp2a_norm += std::norm(std::get<0>(term));
                        }
                        for (const auto& term : temp2b.terms()){
                            temp2b_norm += std::norm(std::get<0>(term));
                        }
                        temp2a.mult_coeffs(std::sqrt(2.0/temp2a_norm));
                        temp2b.mult_coeffs(std::sqrt(2.0/temp2b_norm));

                        if(temp2a.terms().size() > 0){
                            add_term(1.0, temp2a);
                        }
                        if(temp2b.terms().size() > 0){
                            add_term(1.0, temp2b);
                        }
                    }
                }
            }
        }
    } else {
        throw std::invalid_argument( "Invalid pool_type specified." );
    }
}

void SQOpPool::fill_pool_kUpCCGSD(int kmax)
{
    for(int k=0; k < kmax; ++k){
        size_t norb = nocc_ + nvir_;
        for(size_t i=0; i<norb; i++){
            size_t ia = 2*i;
            size_t ib = 2*i+1;
            for (size_t a=i; a<norb; a++){
                size_t aa = 2*a;
                size_t ab = 2*a+1;

                if( aa != ia ){
                    SQOperator temp1a;
                    temp1a.add_term(+1.0, {aa}, {ia});
                    temp1a.add_term(-1.0, {ia}, {aa});
                    temp1a.simplify();
                    if(temp1a.terms().size() > 0){
                        add_term(1.0, temp1a);
                    }
                }

                if( ab != ib ){
                    SQOperator temp1b;
                    temp1b.add_term(+1.0, {ab}, {ib});
                    temp1b.add_term(-1.0, {ib}, {ab});
                    temp1b.simplify();
                    if(temp1b.terms().size() > 0){
                        add_term(1.0, temp1b);
                    }
                }
            }
        }

        std::vector< std::vector<size_t> > uniqe_2bdy;
        std::vector< std::vector<size_t> > adjnt_2bdy;

        for(size_t p=0; p<norb; ++p){
            size_t pa = 2 * p;
            size_t pb = 2 * p + 1;
            for(size_t q=0; q<norb; ++q){
                size_t qa = 2 * q;
                size_t qb = 2 * q + 1;

                if((pa != qa) && (pb != qb)){
                    SQOperator temp2abab;
                    temp2abab.add_term(-1.0, {pa,pb}, {qa,qb});
                    temp2abab.add_term(+1.0, {qb,qa}, {pb,pa});
                    temp2abab.simplify();
                    if(temp2abab.terms().size() > 0){
                        std::vector<size_t> vtemp {std::get<1>(temp2abab.terms()[0])[0], std::get<1>(temp2abab.terms()[0])[1], std::get<2>(temp2abab.terms()[0])[0], std::get<2>(temp2abab.terms()[0])[1]};
                        std::vector<size_t> vadjt {std::get<1>(temp2abab.terms()[1])[0], std::get<1>(temp2abab.terms()[1])[1], std::get<2>(temp2abab.terms()[1])[0], std::get<2>(temp2abab.terms()[1])[1]};
                        if( (std::find(uniqe_2bdy.begin(), uniqe_2bdy.end(), vtemp) == uniqe_2bdy.end()) ){
                            if( (std::find(adjnt_2bdy.begin(), adjnt_2bdy.end(), vtemp) == adjnt_2bdy.end()) ){
                                uniqe_2bdy.push_back(vtemp);
                                adjnt_2bdy.push_back(vadjt);
                                add_term(1.0, temp2abab);
                            }
                        }
                    }
                }
            }
        }        
    }
}

void SQOpPool::fill_pool_sq_hva(std::complex<double> coeff, const SQOperator& sq_op){
    std::vector<std::pair< std::vector<size_t>, std::vector<size_t>>> h_vec;
    std::vector<std::pair< std::vector<size_t>, std::vector<size_t>>> hd_vec;

    for (size_t l = 0; l < sq_op.terms().size(); l++){
        std::pair< std::vector<size_t>, std::vector<size_t>> h;
        std::pair< std::vector<size_t>, std::vector<size_t>> hd;

        std::complex<double> hl = std::get<0>(sq_op.terms()[l]);
        h.first  = std::get<1>(sq_op.terms()[l]);
        h.second = std::get<2>(sq_op.terms()[l]);

        // skip the scalar term.
        if(h.first.size()==0 and h.second.size()==0){
            continue;
        }

        hd.first = h.second;
        hd.second = h.first;

        std::reverse(hd.first.begin(), hd.first.end());
        std::reverse(hd.second.begin(), hd.second.end());

        std::sort(h.first.begin(), h.first.end());
        std::sort(h.second.begin(), h.second.end());

        std::sort(hd.first.begin(), hd.first.end());
        std::sort(hd.second.begin(), hd.second.end());
        
        // Determine if term is in current set of terms or term adjoints
        // if it isn't found in either then append the vectors
        if (std::find(h_vec.begin(), h_vec.end(), h) == h_vec.end()){
            if (std::find(hd_vec.begin(), hd_vec.end(), h) == hd_vec.end()){
                SQOperator temp;
                if(h == hd or h.first == h.second){
                    // (Nick) Need this checked out for sure
                    temp.add_term(0.5, h.first,  h.second);
                    temp.add_term(0.5, hd.first, hd.second);
                } else {
                    temp.add_term(1.0, h.first,  h.second);
                    temp.add_term(1.0, hd.first, hd.second);
                    temp.simplify();
                }

                terms_.push_back(std::make_pair(coeff, temp));

                h_vec.push_back(h);
                hd_vec.push_back(hd);

            }
        } 
    }
}

void SQOpPool::fill_pool_df_trotter(
    const DFHamiltonian& df_ham,
    const std::complex<double> coeff)
{
    // structure should resembel the apply DFHam funciton in fci_computer.cc
    size_t nleaves = df_ham.get_trotter_basis_change_matrices().size();

    if (nleaves - 1 != df_ham.get_scaled_density_density_matrices().size()){
        throw std::invalid_argument("Incompatiable array lengths.");
    }

    // NOTE(Nick): the first DF givens rotation should ALREADY be time scalled 
    // by the user, otherwise this routine won't match the trotter evolution.
    // As such we anticipate the resulting pool will be evolved by coeff=dt,
    // and so this will correctly scale the coefficents. 
    append_givens_ops_sector(
        df_ham.get_trotter_basis_change_matrices()[0],
        1.0/coeff,
        true
    );

    append_givens_ops_sector(
        df_ham.get_trotter_basis_change_matrices()[0],
        1.0/coeff,
        false
    );

    for (size_t l = 1; l < nleaves; ++l) {
        append_diagonal_ops_all(
            df_ham.get_scaled_density_density_matrices()[l - 1],
            1.0
        );

        // NOTE(Nick): subsequent givens rotations DON'T need time scaling
        append_givens_ops_sector(
            df_ham.get_trotter_basis_change_matrices()[l],
            1.0/coeff,
            true
        );

        append_givens_ops_sector(
            df_ham.get_trotter_basis_change_matrices()[l],
            1.0/coeff,
            false
        );
    }
}

void SQOpPool::append_givens_ops_sector(
    const Tensor& U,
    const std::complex<double> coeff,
    const bool is_alfa)
{
    size_t sigma = 0;
    if (is_alfa){ 
        sigma = 0; 
    } else {
        sigma = 1;
    }

    U.square_error();
    Tensor U2 = U;

    //NOTE(Nick): May be SLOW, or don't need to compute, could just store rots_and_diag
    // in DFHamiltonain class pass directly.
    auto rots_and_diag = DFHamiltonian::givens_decomposition_square(U2);

    auto ivec = std::get<0>(rots_and_diag);
    auto jvec = std::get<1>(rots_and_diag);
    auto thts = std::get<2>(rots_and_diag);
    auto phis = std::get<3>(rots_and_diag);

    auto diags = std::get<4>(rots_and_diag);

    for (size_t k = 0; k < ivec.size(); k++){
        size_t i = ivec[k];
        size_t j = jvec[k];
        double tht = thts[k];
        double phi = phis[k];

        if (std::abs(phi) > 1.0e-12){
            SQOperator num_op1; 
            num_op1.add_term(-phi/2.0, {2 * j + sigma}, {2 * j + sigma});
            num_op1.add_term(-phi/2.0, {2 * j + sigma}, {2 * j + sigma});   
            terms_.push_back(std::make_pair(coeff, num_op1));      
        }

        if (std::abs(tht) > 1.0e-12) {
            std::complex<double> itheta(0.0, tht);
            SQOperator single;
            single.add_term(-itheta, {2 * i + sigma}, {2 * j + sigma});
            single.add_term(+itheta, {2 * j + sigma}, {2 * i + sigma});
            terms_.push_back(std::make_pair(coeff, single));  
        }
    }
        
    for (size_t l = 0; l < diags.size(); l++){
        if (std::abs(diags[l]) > 1.0e-12) {
            double diag_angle = std::atan2(diags[l].imag(), diags[l].real());
            SQOperator num_op2;
            num_op2.add_term(-diag_angle/2.0, {2 * l + sigma}, {2 * l + sigma});
            num_op2.add_term(-diag_angle/2.0, {2 * l + sigma}, {2 * l + sigma});
            terms_.push_back(std::make_pair(coeff, num_op2));  
        }
    }
}

void SQOpPool::append_diagonal_ops_all(
    const Tensor& V, 
    const std::complex<double> coeff)
{
    V.square_error();

    int norbs = V.shape()[0];

    for (size_t p = 0; p < norbs; p++) {
        for (size_t q = 0; q < norbs; q++) {
            for (size_t sig = 0; sig < 2; sig++){
                for (size_t tau = 0; tau < 2; tau++){
                    size_t pq = p*norbs + q;
                    std::complex<double> vpq = V.read_data()[pq];
                    if (std::abs(vpq) > 1.0e-12){
                        SQOperator num_op;
                        if(2 * p + sig == 2 * q + tau && 2 * q + tau == 2 * p + sig && 2 * p + sig == 2 * q + tau){
                            num_op.add_term(0.5 * vpq, {2 * p + sig}, {2 * p + sig});
                            num_op.add_term(0.5 * vpq, {2 * p + sig}, {2 * p + sig});
                        } else {
                            num_op.add_term(-0.5 * vpq, {2 * p + sig, 2 * q + tau}, {2 * p + sig, 2 * q + tau});
                            num_op.add_term(-0.5 * vpq, {2 * p + sig, 2 * q + tau}, {2 * p + sig, 2 * q + tau});
                        }
                        terms_.push_back(std::make_pair(coeff, num_op));    
                    }
                }
            }
        }
    }
}

std::string SQOpPool::str() const{
    std::vector<std::string> s;
    s.push_back("");
    int counter = 0;
    for (const auto& term : terms_) {
        s.push_back("----->");
        s.push_back(std::to_string(counter));
        s.push_back("<-----\n");
        s.push_back(to_string(term.first));
        s.push_back("[\n");
        s.push_back(term.second.str());
        s.push_back("]\n\n");
        counter++;
    }
    return join(s, " ");
}
