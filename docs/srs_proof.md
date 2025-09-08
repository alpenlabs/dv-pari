### SRS Definition

Components of SRS for QAP are $G_M, G_Q, G_{K,i}$ given in [r1cs prover document](./r1cs_prover.md)

### Toxic Waste Commitment

SRSProver will additionally commit to the toxic waste on a a separate generator H

$H_\tau = \tau H \qquad H_\delta = \delta H \qquad H_\epsilon = \epsilon H $

### Direct Proof

#### Step 1:

SRSProver commits to SRS $G_M, G_Q, G_{K,i}$

#### Step 2:

SRSVerifier samples random witness $(w, q)$ and computes the commitment

$P = \sum_{i} w_i G_{M,i} + \sum_{j} q_j G_{Q,j}$

If SRS is correct, then P represents a commitment of the form

$P = \big( (a(\tau) + b(\tau) \delta + r(\tau) \delta^2) \epsilon \big) G $

SRSVerifier, now, samples a random challenge $\alpha $and also computes the commitment 

$K = \sum_{i \in [m]} k_{a,i} G_{K, 0, i} + k_{b,i} G_{K, 1, i} + \sum_{i \in [2 m]} k_{r,i} G_{K, 2, i}. $

SRS Verifier sends the following to SRSProver:

Proof $\pi$ = $(P, K, a_0, b_0)$ and $\alpha$

Note that $(w, q)$has to satisfy the constraints baked in $G_M, G_Q, G_K$. Therefore $(w, q)$ should be values that satisfy the embedded R1CS constraints -- not just any random number.

#### Step 3

SRSProver can construct a proof of knowledge of toxic waste using standard sigma protocol techniques that

$P - (a + \delta b + \delta^2 r) \epsilon G = (\tau - \alpha) \epsilon K $

i.e. for the relation

$R = \left\{ (\tau, \delta, \epsilon ; H_\tau, H_\delta, H_\epsilon, \pi) : \begin{array}{} H_x = x H \text{ for } x = \tau, \delta, \epsilon \\ P - (a + \delta b + \delta^2 r) \epsilon G = (\tau - \alpha) \epsilon K \end{array} \right\}$

If the SRS is incorrect, SRSProver should not be able to show this relation.

### Sigma Protocol

#### Step 4

SRSProver samples $r_\tau, r_\delta, r_\epsilon $. Let $\hat{x}(Y)$ = $r_x + Y x$ for $x$ = $\tau, \delta, \epsilon $

Define $R_0, R_1$ and $R_2$ by grouping terms

$R_0 + R_1 Y + R_2 Y^2 = Y^2 P - \big( (a Y + b \hat{\delta}(Y)) \hat{\epsilon}(Y) \big) G - (\hat{\tau}(Y) - \alpha Y) \hat{\epsilon}(Y) K$

SRSProver sends $R_0, R_1, R_2, R_\tau = r_\tau H,R_\delta = r_\delta H,R_\epsilon = r_\epsilon H $

#### Step 5

SRSVerifier samples $\chi$

#### Step 6

SRSProver sends $x' = r_x + \chi x$ = $\hat{x}(\chi)$ for $x = \tau, \delta, \epsilon $


#### Step 7

SRSVerifier checks

$x' H = R_x + \chi H_x \text{ for } x = \tau, \delta, \epsilon $

$R_0 + \chi R_1 + \chi^2 R_2 = \chi^2 P - \big( (a \chi + \delta' b + \delta'^2 r) \epsilon' \big) G - (\tau' - \alpha \chi) \epsilon K $