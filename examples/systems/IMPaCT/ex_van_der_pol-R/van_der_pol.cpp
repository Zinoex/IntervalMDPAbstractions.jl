// Run:   make
//        ./van_der_pol

#include <iostream>
#include <vector>
#include <functional>
#include "../../src/IMDP.h"
#include <armadillo>
#include <chrono>

using namespace std;
using namespace arma;

/*
 ################################# PARAMETERS ###############################################
 */

// Set the dimensions
const int dim_x = 2;
const int dim_u = 1;
const int dim_w = 0;

// Define lower bounds, upper bounds, and step sizes
// States
const vec ss_lb = {-3.92, -3.92};
const vec ss_ub = {3.92, 3.92};
const vec ss_eta = {0.16, 0.16};
// Inputs
const vec is_lb = {-1};
const vec is_ub = {1};
const vec is_eta = {0.2};

//standard deviation of each dimension
const vec sigma = {sqrt(0.2), sqrt(0.2)};

// logical expression for target region and avoid region
auto target_condition = [](const vec& ss) { return (ss[0] >= -1.32 && ss[0] <= -0.78) && (ss[1] >= -2.82 && ss[1] <= -2.08); };

//dynamics - 2 parameters
auto dynamics = [](const vec& x, const vec& u) -> vec {
    float sampling_time = 0.1; 

    vec xx(dim_x);
    xx[0] = x[1] + x[2] * sampling_time;
    xx[1] = x[2] + (-x[1] + (1 - x[1]) * (1 - x[1]) * x[2]) * sampling_time + u[1];
    return xx;
};

/*
 ################################# MAIN FUNCTION ##############################################
 */

int main() {
    
    /* ###### create IMDP object ###### */
    IMDP mdp(dim_x, dim_u, dim_w);
    
    /* ###### create finite sets for the different spaces ###### */
    mdp.setStateSpace(ss_lb, ss_ub, ss_eta);
    mdp.setInputSpace(is_lb, is_ub, is_eta);
    
    /* ###### relabel states based on specification ###### */
    mdp.setTargetSpace(target_condition, true);
    
    /*###### save the files ######*/
    mdp.saveStateSpace();
    mdp.saveInputSpace();
    mdp.saveTargetSpace();
    
    /*###### set dynamics and noise ######*/
    mdp.setDynamics(dynamics);
    mdp.setNoise(NoiseType::NORMAL);
    mdp.setStdDev(sigma);
    
    /* ###### calculate abstraction for target vectors ######*/
    mdp.targetTransitionVectorBounds();
    
    /* ###### save target vectors ######*/
    mdp.saveMinTargetTransitionVector();
    mdp.saveMaxTargetTransitionVector();
    
    /* ###### calculate abstraction for avoid vectors ######*/
    mdp.minAvoidTransitionVector();
    mdp.maxAvoidTransitionVector();
    
    /* ###### save avoid vectors ######*/
    mdp.saveMinAvoidTransitionVector();
    mdp.saveMaxAvoidTransitionVector();
    
    
    /* ###### calculate abstraction for transition matrices ######*/
    mdp.transitionMatrixBounds();
    
    /* ###### save transition matrices ######*/
    mdp.saveMinTransitionMatrix();
    mdp.saveMaxTransitionMatrix();
    
    /* ###### synthesize infinite horizon controller (true = pessimistic, false = optimistic) ######*/
    // mdp.infiniteHorizonReachController(true);
    
    /* ###### synthesize finite horizon controller (true = pessimistic, false = optimistic) ######*/
    mdp.finiteHorizonReachControllerSorted(true, 10);
    
    /* ###### save controller ######*/
    mdp.saveController();
    
    
    return 0;
}

