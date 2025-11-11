#ifndef RLPOLICY_H
#define RLPOLICY_H

# ifdef __cplusplus
extern "C" {
# endif // ifdef __cplusplus

#include "osqp.h"

void *rl_policy_load(const char* module_path);
int rl_policy_unload(void *policy);
    
/**
 * Update the rho vector based on a trained RL policy.
 * @params work Workspace
 * @return Exitflag
 */
int rl_policy_compute_vec(OSQPWorkspace* work);
    
# ifdef __cplusplus
}
# endif // ifdef __cplusplus

#endif