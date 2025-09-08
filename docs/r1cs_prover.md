Let $L_i$ and $L_i'$ be the lagranges for the domains $\mathcal{D}$ and $\mathcal{D'}$ respectively. Let $L''_i$ be the lagranges for the combined domain $\mathcal{D}'' = \mathcal{D} \cup \mathcal{D}'$.

$A w \circ B w = C w + D x$

$\big( m_j(X, Y) \big)_{j \in [n]} = (A + Y B + Y^2 C)^\top \big( L_i(X) \big)_{i \in [m]}$

$G_M = \big(  m_j(\tau, \delta) \epsilon G \big)_{j \in [n]}$

$G_{Q} = \big( z(\tau) L'_i(\tau) \delta^2 \epsilon  G \big)_{i \in [m]}$

$G_{K,k} = \big( L_i(\tau) \delta^k  G \big)_{i \in [m]} \text{ for } k = 0,1$

$G_{K,2} = \big( L''_i(\tau)\delta^2 G \big)_{i \in [2 m]}$

$i(X) = \sum_{i \in [k]} x_i X^{i-1} \qquad i(X)|_\mathcal{D} = D x$

$a(X) |_\mathcal{D} = A w \qquad b(X) |_\mathcal{D} = B w \qquad c(X) |_\mathcal{D} = C w$.

$q(X) = \frac{a(X) b(X) - c(X) - i(X)}{z(X)}.$

$P = \sum_{j \in [n]} w_j G_{M, j} + \sum_{i \in [m]} q_i G_{Q, i}.$

$ r(X) = c(X) + z(X) q(X) = a(X) b(X) - i(X) $

$ P = \big( (a(\tau) + b(\tau) \delta + r(\tau) \delta^2) \epsilon \big) G $

$ a_0 = \text{Eval}_\mathcal{D}(a(X)|_\mathcal{D}, \alpha) \qquad b_0 = \text{Eval}_\mathcal{D}(b(X)|_\mathcal{D}, \alpha) $

$ \left( k_p(X) = \frac{p(X) - p(\alpha)}{X - \alpha} \right)_{p = a,b, r} $

$ Q = \sum_{i \in [m]} k_{a,i} G_{K, 0, i} + k_{b,i} G_{K, 1, i} + \sum_{i \in [2 m]} k_{r,i} G_{K, 2, i}. $

Proof $\pi = (P, Q, a_0, b_0)$

Verifier:

$ \alpha := h(\sigma, P) $

$ r_0  := a_0 b_0 - \sum_{i \in [k]} x_i \alpha^{i-1} $

$u_0 := (a_0 + \delta b_0 + \delta^2 r_0) \epsilon $

$v_0 := (\tau - \alpha) \epsilon  $

$ P \overset{?}{=} v_0 K + u_0 G. $

Sigma Protocol

1. Alice samples $ r_\tau, r_\delta, r_\epsilon $. Let $ \hat{x}(Y) = r_x + Y x $ for $ x = \tau, \delta, \epsilon $

2. Define $ R_0 $, $ R_1 $ and $ R_2 $ by grouping terms

$R_0 + R_1 Y + R_2 Y^2 = Y^2 P - \big( (a Y + b \hat{\delta}(Y)) \hat{\epsilon}(Y) \big) G - (\hat{\tau}(Y) - \alpha Y) \hat{\epsilon}(Y) Q$

3. Alice sends $ R_0, R_1, R_2, R_\tau = r_\tau H,R_\delta = r_\delta H,R_\epsilon = r_\epsilon H $

4. Bob samples $\chi$

5. Alice sends $ x' = r_x + \chi x = \hat{x}(\chi) $ for $ x = \tau, \delta, \epsilon $ 

6. Bob checks

$x' H = R_x + \chi H_x \text{ for } x = \tau, \delta, \epsilon $

$R_0 + \chi R_1 + \chi^2 R_2 = \chi^2 P - \big( (a \chi + \delta' b + \delta'^2 r) \epsilon' \big) G - (\tau' - \alpha \chi) \epsilon Q $