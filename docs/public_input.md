We introduce a new variant of **R1CS** that supports more-efficient input-polynomial evaluation.  
Here we describe a compiler that turns a regular R1CS into these forms.

* * *

#### From standard R1CS…

Suppose we have an R1CS instance for a relation R\mathcal{R}R encoded by three matrices

$A,B \in F^{m×n}, C ∈ F^{m×(n+k)}$

such that $(x,w)∈R$ iff

$Aw∘Bw  =  C⁣(x ∥ w)$ 

This is the **standard** representation used by Groth-16.

* * *

#### …to separating public and private columns

First, decompose $\mathbb{C}$ into blocks

$U \in \mathbb{F}^{m\times k}, \qquad V \in \mathbb{F}^{m\times n}$

corresponding to columns that act on $x$ and $w$ respectively, so that

$C(x∥w) = Ux+Vw$

To pass $x$ more efficiently as public input, fix a matrix

$D ∈ F^{m×k}, D_{ij} = d_i^j$

which encodes the evaluations of the input polynomial over the domain.

* * *

#### Defining the new relation

Let $n' =k + n$ and set $w'=x∥w$.

Define a new relation $R'⊂F^k×F^{n'}$ by

$(x,w)∈R    ⟺    (x,w')∈R'$

Construct new matrices

$A'w' = A w$

$B'w'=Bw$

$C'w'=(U−D) x+Vw$

* * *

#### Resulting SR1CS form

Hence

$(x,w)∈R    ⟺    (x,w')∈R'    ⟺    A'w'  ∘  B'w'  =  C'w'+Dx$

This completes the compilation from the standard R1CS representation to the new SR1CS variant with explicit handling of public inputs.