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

$ K = \sum_{i \in [m]} k_{a,i} G_{K, 0, i} + k_{b,i} G_{K, 1, i} + \sum_{i \in [2 m]} k_{r,i} G_{K, 2, i}. $

Proof $\pi = (P, K, a_0, b_0)$

Verifier:

$ \alpha := h(\sigma, P) $

$ r_0  := a_0 b_0 - \sum_{i \in [k]} x_i \alpha^{i-1} $

$u_0 := (a_0 + \delta b_0 + \delta^2 r_0) \epsilon $

$v_0 := (\tau - \alpha) \epsilon  $

$ P \overset{?}{=} v_0 K + u_0 G. $