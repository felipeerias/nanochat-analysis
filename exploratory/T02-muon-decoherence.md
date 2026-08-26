# T02 — finite-step Polar Express as a rounding amplifier

Status: **draft**.

## Phenomenon

The production Muon optimizer and the versioned eager reference in
[telemetry.py](../../nanochat/nanochat/telemetry.py) implement the same nominal
algebra, but not the same floating-point map. In compiled bf16, their applied
updates differ by roughly 3–10% per matrix in update-relative L2. The median
is about 3–5% on real mid-training gradients; random-input calibration reaches
18.9% on the smallest matrices. Forcing both paths to fp32 sends most matrices
to roundoff, but leaves a worst tail near $3.3\times10^{-3}$. Compiling the
reference does not remove the discrepancy because it produces another fusion
and another set of rounding points, not the production kernel's intermediate
values.

[I0003](../investigations/0003-decoherence-vs-depth/conclusion.md)
finds two pieces of structure that a model must address:

- median <code>muon/replay_update_relerr</code> falls by 6.6% at d14 and 11.2%
  at d16 relative to d12, although width and depth co-vary;
- parameter role explains 80–88% of within-checkpoint variation, while depth
  position explains almost none. The stable ordering is <code>mlp_out</code>,
  <code>attn_k</code>, <code>attn_q</code>/<code>mlp_in</code>,
  <code>attn_out</code>, <code>attn_v</code>, <code>ve_gate</code>, spanning
  about $0.84\times$ to $2.08\times$ each depth's median.

[I0002](../investigations/0002-bf16-vs-fp32-curvature/conclusion.md)
is a warning against a generic “bf16 is noisy” explanation: bf16 barely
changes the main curvature values while destroying cancellation-sensitive
validation quantities. Decoherence likewise needs a map-specific condition
number. It is not evidence that every bf16 quantity is wrong by a few percent.

The question here is narrower than the causal question in
[E05](../experiments/E05-decoherence-intervention.md): **why can five Polar
Express stages turn different rounding placement into percent-level update
angles, and what matrix properties control the amplification?** This note does
not say whether the difference matters to training.

## 1. The source-exact map

Consider one $m\times n$ Nesterov matrix $G$, and let
$q=\min(m,n)$. Suppress stacking because production and telemetry apply the
same matrix operations independently to each stack slice.

The implementation in [optim.py](../../nanochat/nanochat/optim.py) is:

1. form the Nesterov matrix in the parameter dtype and cast the working matrix
   to bf16;
2. apply MuonEq, scaling every row to
   $t=\lVert G\rVert_F/\sqrt m$, subject to the $10^{-6}$ row-norm clamp;
3. divide by $1.01\lVert\cdot\rVert_F+10^{-6}$;
4. apply five Polar Express stages;
5. cast back to the parameter dtype and snap the Frobenius norm to $\sqrt q$;
6. apply a norm-preserving factored second-moment scale;
7. add cautious weight decay and apply the shape-adjusted learning rate.

Write $Z=\operatorname{Eq}(G)$ and

$$
X_0 = \frac{Z}{1.01\lVert Z\rVert_F+10^{-6}}.
$$

In the corresponding exact-arithmetic map and away from clamped rows, MuonEq
preserves the Frobenius norm: its $m$ output rows all have norm
$\lVert G\rVert_F/\sqrt m$. Consequently

$$
\lVert X_0\rVert_F < \alpha := \frac1{1.01}=0.990099\ldots,
\qquad 0\leq \sigma_i(X_0)<\alpha.
$$

The relevant spectrum is therefore the spectrum **after Nesterov momentum and
MuonEq**, not the raw-gradient spectrum. A row near the clamp is a separate
source of bad conditioning that the model below folds into its initial
perturbation $E_0$.

For either the tall or wide branch, exact arithmetic preserves singular
vectors. If $X_{j-1}=U\operatorname{diag}(s_{j-1,i})V^T$, stage $j$ maps
each singular value through

$$
f_j(s)=a_js+b_js^3+c_js^5,
\qquad s_{j,i}=f_j(s_{j-1,i}).
$$

The source coefficients are:

| $j$ | $a_j$ | $b_j$ | $c_j$ |
|---:|---:|---:|---:|
| 1 | 8.156554525 | -22.483292926 | 15.878769915 |
| 2 | 4.042929935 | -2.808917466 | 0.500017845 |
| 3 | 3.891667802 | -2.772484153 | 0.506064818 |
| 4 | 3.285753658 | -2.368129493 | 0.464490242 |
| 5 | 2.346541326 | -1.709782838 | 0.423235512 |

Thus the five-stage scalar map is the fixed odd polynomial

$$
p=f_5\circ f_4\circ f_3\circ f_2\circ f_1,
$$

of degree $5^5=3125$. This is not five repetitions of a stationary
Newton–Schulz map, and the source explicitly does not iterate it to a fixed
point. “Distance to a fixed point” is therefore only an analogy. The exact
local object is the derivative of this finite composition.

## 2. Scalar and matrix sensitivity

For a singular value $s$, define its exact stage orbit by $s_0=s$ and
$s_j=f_j(s_{j-1})$. A radial perturbation has gain

$$
p'(s)=\prod_{j=1}^5
\left(a_j+3b_js_{j-1}^2+5c_js_{j-1}^4\right).
$$

At zero this is exact:

$$
p'(0)=\prod_{j=1}^5a_j=989.4683957.
$$

The downstream zero-mode gains for an additive error placed at different
points are:

| error injected | downstream gain near $s=0$ |
|---|---:|
| before stage 1 | 989.468 |
| after stage 1 | 121.310 |
| after stage 2 | 30.005 |
| after stage 3 | 7.710 |
| after stage 4 | 2.347 |
| after stage 5 | 1 |

This is the first answer to why rounding *placement* matters. Two executions
can agree to one rounding unit after an early matrix product and still differ
by tens or hundreds of those units after the remaining stages. Moving the
same rounding event past a stage changes its downstream gain.

There is also a conservative coefficient-only bound. Maximizing each quintic
and its derivative over a reachable singular-value interval gives successive
absolute bounds

$$
(M_0,\ldots,M_5)
=(0.99010,1.99169,1.96637,1.86922,1.57554,1.14113)
$$

and

$$
L_j=\sup_{|s|\leq M_{j-1}}|f'_j(s)|
=(18.3317,9.9558,9.5612,6.8152,2.6535).
$$

Therefore

$$
\operatorname{Lip}(p;[-\alpha,\alpha])
\leq \prod_jL_j = 3.156\times10^4. \tag{1}
$$

The bound is intentionally loose because the five worst derivative locations
cannot generally lie on one scalar orbit. Dense numerical evaluation of the
displayed one-dimensional recurrence suggests a much tighter maximum radial
gain of about 989.47, at zero. Equation (1), unlike that numerical envelope,
is a safe stagewise bound and is useful for the fp32 tail:
$2^{-24}\times3.16\times10^4=1.88\times10^{-3}$ for one early perturbation,
before adding the other rounding sites. A several-$10^{-3}$ worst tail is
therefore allowed even when most fp32 matrices agree to roundoff.

The scalar derivative is not the whole matrix condition number. Let
$X_0=U\Sigma V^T$, with positive singular values $\sigma_i$. In the singular
basis, the Fréchet derivative of the spectral map
$P(X)=U\operatorname{diag}(p(\sigma_i))V^T$ multiplies:

- a diagonal/radial component by $p'(\sigma_i)$;
- a symmetric $i,j$ component by

  $$
  \ell^-_{ij}=
  \frac{p(\sigma_i)-p(\sigma_j)}{\sigma_i-\sigma_j};
  $$

- an antisymmetric $i,j$ component by

  $$
  \ell^+_{ij}=
  \frac{p(\sigma_i)+p(\sigma_j)}{\sigma_i+\sigma_j};
  $$

- a component coupling a singular direction to a rectangular null space by
  $p(\sigma_i)/\sigma_i$.

Because $p$ is odd, all of these divided differences are bounded by the
maximum derivative of $p$ on the signed spectral interval. Near zero,

$$
p'(\sigma)\to989.47,
\qquad \frac{p(\sigma)}{\sigma}\to989.47. \tag{2}
$$

Equation (2) is the important distinction between radial and angular
sensitivity. For a relative radial perturbation,

$$
\kappa_{\rm rad}(\sigma)
=\left|\frac{\sigma p'(\sigma)}{p(\sigma)}\right|\to1
\quad(\sigma\to0),
$$

but a small absolute perturbation that rotates or mixes a near-null singular
direction sees the much larger gain $p(\sigma)/\sigma$. A spectrum
concentrated in a flat part of the finite polynomial is relatively
insensitive. A spectrum with mass near zero, or spread across steep and
oscillatory parts of $p$, is sensitive. Merely saying “the singular values
are small” is insufficient: the perturbation's orientation in the singular
basis matters too.

This also corrects the limiting sign-function analogy. A converged polar
factor has zero radial derivative at positive singular values but becomes
arbitrarily sensitive to rotations as the smallest singular value approaches
zero. Five-step Polar Express replaces that singularity by a finite but large
gain near 989, while retaining nonzero, oscillatory radial derivatives away
from zero.

## 3. Rounding-placement model

Let the eager and production executions first differ by $E_0$ at $X_0$,
and let $\Xi_j$ be the difference between their local rounding errors at
stage $j$. To first order,

$$
E_j=D\Phi_j[X_{j-1}]E_{j-1}+\Xi_j,
$$

so

$$
E_5=DP[X_0]E_0+
\sum_{j=1}^5D(\Phi_5\circ\cdots\circ\Phi_{j+1})[X_j]\,\Xi_j
+O(E^2). \tag{3}
$$

The $\Xi_j$ include different association of $X X^T$ or $X^T X$, the
second matrix product, scalar-polynomial evaluation, casts, and fused versus
materialized intermediates. Equation (3) explains why compiling the eager
reference need not help: it changes the $\Xi_j$, but it does not force them
to equal the production kernel's.

Bfloat16 unit roundoff is $u_{\rm bf16}=2^{-8}=0.00390625$. A modest effective
condition number of 2–10 already turns one rounding-scale perturbation into
0.8–4% error; several stage-local sources can reach the observed 3–10%.
The gains in (1)–(3) explain the long tail without claiming that every matrix
attains the worst bound. The absolute error level cannot be predicted until
the covariance and orientation of the $\Xi_j$ are specified.

## 4. Why the final update error can be smaller

### Muon+ Frobenius snap

Let $Y=P(X_0)$. Muon+ applies

$$
R(Y)=\sqrt q\frac{Y}{\lVert Y\rVert_F}.
$$

Its first variation is

$$
DR_Y[E]=\frac{\sqrt q}{\lVert Y\rVert_F}
\left(E-Y\frac{\langle Y,E\rangle}{\lVert Y\rVert_F^2}\right). \tag{4}
$$

Thus the radial component of the post-polar discrepancy is removed exactly
to first order. In relative units,

$$
\frac{\lVert DR_Y[E]\rVert_F}{\lVert R(Y)\rVert_F}
=\frac{\lVert E_\perp\rVert_F}{\lVert Y\rVert_F}
\leq\frac{\lVert E\rVert_F}{\lVert Y\rVert_F}. \tag{5}
$$

The snap suppresses; it does not amplify first-order relative error. It also
means <code>muon/norm_deviation</code> can remain at roundoff while
<code>muon/replay_update_relerr</code> is several percent. Norm conformance
is not a coherence check.

If no later anisotropic scale or decay were present, two snapped directions
of equal norm would obey the exact relation

$$
\frac{\lVert u_a-u_r\rVert_F}{\lVert u_a\rVert_F}
=\sqrt{2(1-\cos(u_a,u_r))}. \tag{6}
$$

The current telemetry does not expose the production post-polar intermediate,
so (6) is a prediction for a future stage-paired diagnostic rather than a
quantity recoverable from the existing stage cosines.

### Factored second moment

Let $D$ be the broadcast diagonal operator formed by the reciprocal square
root of the updated second-moment buffer. The source computes

$$
F_D(g)=\frac{\lVert g\rVert_F}{\lVert Dg\rVert_F}Dg. \tag{7}
$$

This identity follows directly from <code>v_norm</code> and
<code>v_norm_new</code> in the code. Hence
$\lVert F_D(g)\rVert_F=\lVert g\rVert_F=\sqrt q$: factored scaling is another
norm-preserving direction map. For fixed $D$, its angular condition number is
at most

$$
\kappa(D)=\frac{\max_iD_i}{\min_iD_i}.
$$

It is nearly neutral when the factors are uniform, but it can amplify an
angular discrepancy when they are dispersed. In production $D$ also changes
with $g$ through the new squared mean, at weight $1-\beta_2=0.1$; this
feedback is small for a well-established, unclamped buffer but can be large
near the $10^{-10}$ clamp. The recorded
<code>muon/factored_scale_dispersion</code> is a useful proxy, not a bound on
$\kappa(D)$, because a coefficient of variation does not determine the
extrema.

### Cautious decay and the measurement denominator

With a fixed cautious mask, write the data and decay parts as $d=\eta u$ and
$c=\eta\,wd\,\theta\odot M$. The mask makes
$\langle d,c\rangle\geq0$. If

$$
\rho=\frac{\lVert c\rVert}{\lVert d\rVert},\qquad
\chi=\cos(d,c),
$$

then a data-direction discrepancy is diluted in the applied-update metric by

$$
\frac{\lVert d\rVert}{\lVert d+c\rVert}
=\frac1{\sqrt{1+\rho^2+2\rho\chi}}\leq1. \tag{8}
$$

All terms in (8) are represented by the recorded
<code>muon/data_norm</code>, <code>muon/decay_norm</code>, and
<code>muon/cos_data_decay</code>. This is a quantitative reason for final
<code>muon/replay_update_relerr</code> to be smaller than a post-polar
relative error.

The exception is mask disagreement. For two paths,

$$
\delta\Delta=\eta\,\delta u+
\eta\,wd\,\theta\odot(M_a-M_r). \tag{9}
$$

The second term is discontinuous at $u_{ij}\theta_{ij}=0$ and has no useful
derivative bound. <code>cautious_mask_fraction</code> does not record the
margin to that boundary or cross-path mask disagreement, so (9) is an
unmodeled tail source.

## 5. Width: what an MP analogy does and does not predict

For an iid $q\times\ell$ matrix with $q\leq\ell$, row equilibration and
Frobenius normalization suggest the analogy

$$
\sigma=\frac{t}{1.01\sqrt q},
$$

where $t$ has the Marchenko–Pastur singular-value density

$$
h_\gamma(t)=
\frac{\sqrt{\left((1+\sqrt\gamma)^2-t^2\right)
\left(t^2-(1-\sqrt\gamma)^2\right)}}{\pi\gamma t},
\quad \gamma=\frac q\ell, \tag{10}
$$

on $1-\sqrt\gamma\leq t\leq1+\sqrt\gamma$. This is an analogy, not a
derivation for transformer gradients: token correlations, momentum, zero
initializations, and role-specific activation/error covariances violate the
iid assumptions.

A radial relative-perturbation score implied by (10) is

$$
K_{\rm rad}(q,\gamma)^2=
\frac{\mathbb E[(\sigma p'(\sigma))^2]}
     {\mathbb E[p(\sigma)^2]}. \tag{11}
$$

Numerical quadrature of the displayed polynomial and density gives:

| MP shape | $K_{\rm rad}(768)$ | $K_{\rm rad}(896)$ | $K_{\rm rad}(1024)$ | d16/d12 |
|---|---:|---:|---:|---:|
| square, $\gamma=1$ | 2.12 | 2.05 | 2.03 | 0.960 |
| 4:1 rectangular, $\gamma=1/4$ | 2.56 | 2.38 | 2.22 | 0.868 |

The rectangular radial model predicts an exponent near $q^{-0.49}$, but the
square model predicts only $q^{-0.14}$. Including isotropic angular
perturbations through the divided differences weakens both trends further. If
$H$ is the mean squared symmetric/antisymmetric divided difference and
$N=\mathbb E[(p(\sigma)/\sigma)^2]$, the isotropic Fréchet score is

$$
K_{\rm iso}^2=
\frac{\gamma}{q\,\mathbb E[p(\sigma)^2]}
\left[H+(\gamma^{-1}-1)N\right]. \tag{12}
$$

It falls only 1.6% from q=768 to 1024 for square MP and about 1.0% for 4:1 MP.
Therefore **MP spectra plus isotropic relative rounding do not by themselves
predict the observed 11% fall**, especially the 12.3% fall reported for the
square attention shape class. The width trend requires an additional claim
about which modes the compiled/eager rounding difference occupies.

One simple reduced model is that $k=O(1)$ sensitive singular directions carry
$q$-independent discrepancy energy while Muon+ fixes total update energy to
$q$. Then

$$
\epsilon(q):=\operatorname{median}(\text{replay relative error})
\simeq C_r\sqrt{\frac{k_r}{q}}
\propto q^{-1/2}. \tag{13}
$$

For the nanochat widths, (13) predicts

$$
\frac{\epsilon_{896}}{\epsilon_{768}}=0.9258
\quad(-7.42\%),\qquad
\frac{\epsilon_{1024}}{\epsilon_{768}}=0.8660
\quad(-13.40\%). \tag{14}
$$

The observed offsets, -6.6% and -11.2%, imply exponents 0.443 and 0.413
respectively. Their residuals from (14) are monotone in width: 0, +0.8, and
+2.2 percentage points of the d12 level. Both are inside the 3.5% seed floor,
at 0.23 and 0.63 floor units, so this does not reject (13), but (13) does not
predict the residual shape.

A width-independent bulk contribution is possible in the more general
rounding-placement model (3). If the stage-weighted, post-snap tangential
error has $O(q)$ bulk energy in addition to $O(1)$ energy in a bounded number
of sensitive modes, then a natural uncorrelated-components extension is

$$
\epsilon(q)^2\simeq \frac{H_r^2}{q}+C_{b,r}^2. \tag{13a}
$$

Here $C_{b,r}$ is set by the per-direction RMS of the local fusion/rounding
differences $\Xi_j$, after their downstream gains, tangent projection,
factored scaling, and decay dilution. Width independence requires this RMS to
stabilize with $q$; equation (3) does not require that assumption. A linear
form $Aq^{-1/2}+C$ instead requires coherent alignment of the localized and
bulk discrepancies (or another reason their norms add rather than their
squared norms), which the model also does not supply.

For the three reported ratios, the linear form fits $C=0.161$ of the d12
level, with residuals no larger than 0.26 percentage points. At the 4.24%
d12 median this is 0.68 percentage points of relative error, about
$1.7u_{\rm bf16}$, so its magnitude is not numerically implausible. But an
uncorrelated fit of (13a) is comparably good and puts the floor at 0.387 of
the d12 level. The fp32 worst tail of $3.3\times10^{-3}$ establishes that
fusion-dependent errors can survive higher precision after amplification; it
is a worst tail at one width, not a median width scaling, and therefore does
not select either floor or set its value. The honest status is that an
additive floor is a plausible, ad hoc extension of P3, while the monotone
residual remains unexplained by the pure finite-sensitive-mode reduction.

This is also not evidence for the finite-sensitive-mode premise. MP does not
imply that $k$ is constant, and all existing runs set width to $64$ times
depth. Width and depth are therefore functionally dependent, in addition to
the schedule and horizon changes along the size ray.

## 6. Role dependence

The source fixes the matrix shapes on this sweep:

- <code>attn_q</code>, <code>attn_k</code>, <code>attn_v</code>, and
  <code>attn_out</code> are all $d\times d$;
- <code>mlp_in</code> is $4d\times d$, while <code>mlp_out</code> is
  $d\times4d$;
- <code>ve_gate</code> is only $(d/128)\times12$, namely $6\times12$,
  $7\times12$, and $8\times12$ on d12, d14, and d16.

The MP analogy assigns a square hard edge to every attention role, the same
gapped $\gamma=1/4$ law to both MLP roles, and a small-$q$, moderately
rectangular law to <code>ve_gate</code>. It therefore predicts
<code>ve_gate</code> to be vulnerable qualitatively: it has little
self-averaging and the smallest normalization denominator. It can also make
<code>mlp_out</code> low if its actual spectrum resembles the gapped
rectangular law.

It cannot predict the measured full ordering. Shape-only spectra predict the
four attention roles to tie, yet <code>attn_k</code> is near the bottom while
<code>attn_v</code> is near the top. They also predict <code>mlp_in</code>
and <code>mlp_out</code> to tie, yet those roles are widely separated. The
discrepancy can mean either that their real post-Nesterov, post-MuonEq spectra
differ, or that equal spectra receive differently oriented rounding
perturbations and post-polar scaling.

To make a role prediction rather than name a role after observing it, the
model needs, per matrix:

1. singular-value quantiles of $X_0$, especially mass near zero and in the
   high-$|p'|$ bands;
2. a computed Fréchet sensitivity score using $p'$, $\ell^-_{ij}$, and
   $\ell^+_{ij}$;
3. the covariance of production-minus-reference errors at each Polar Express
   stage in the singular basis;
4. extrema of the factored scale and the cautious-mask margin distribution.

Spectra alone determine the available gains, not which gains a structured
rounding error excites. Linear-layer gradient spectra themselves depend on
both activation covariance and back-propagated error covariance, so those
role-specific quantities would be needed for an upstream explanation of the
spectra.

## 7. Predictions and falsification

### P1 — coefficient-level amplification

For a stage-paired replay, errors inserted near a small singular mode should
show downstream ratios close to the zero-mode gains

$$
121.3:30.0:7.71:2.35:1
$$

for injection after stages 1 through 5, until nonlinear terms or bf16
quantization dominate. The full small-mode input gain is 989.47 and the safe
all-spectrum bound is $3.16\times10^4$. Failure of the measured local
Fréchet response to fit the divided-difference formula would refute the
finite-polynomial model itself, rather than merely a distributional
assumption.

### P2 — normalization removes norm error, not angle

Percent-level <code>muon/replay_update_relerr</code> may coexist with
near-zero <code>muon/norm_deviation</code>. With stage-paired
actual/reference directions, the post-snap discrepancy should obey (6), then
change according to factored-scale anisotropy and the decay dilution (8). At
fixed spectral sensitivity, the upper envelope of angular error should grow
with factored-scale anisotropy, while larger decay dilution should reduce the
applied-update relative error. A mask-flip tail is explicitly exempt from this
smooth prediction. The recorded coefficient of variation is only a proxy for
the required anisotropy.

### P3 — conditional width law in the recorded channel

For a fixed-depth width sweep such as
[E03](../experiments/E03-width-vs-decoherence.md), every role with
$q\propto\text{width}$ has the conditional prediction

$$
\frac{\epsilon(Rq)}{\epsilon(q)}=R^{-1/2},
$$

for the role-stratified median <code>muon/replay_update_relerr</code>, measured
after the 400-step momentum ramp and at matched normalized progress. The
constants $C_r\sqrt{k_r}$ may differ by role, but the width exponent should
be $-1/2$. A fitted common exponent near zero, or opposite signs across
adequately replicated roles, would refute this reduced width model. A
systematic flattening with width would reject the pure law and motivate a
precommitted comparison with a bulk-floor extension such as (13a). Neither
result would refute the polynomial amplification in (1)–(3). The tiny
<code>ve_gate</code> matrices are a separate asymptotic regime because their
aspect ratio changes and $q=6$–8.

P3 is a prediction for E03, not a claim supported by the current data. In all
seven schema-v3 sweep runs, <code>aspect_ratio=64</code>, giving widths 768,
896, and 1024 at depths 12, 14, and 16. Thus width is exactly proportional to
depth and $q^{-1/2}$ and $\text{depth}^{-1/2}$ are the same numerical
prediction. No reanalysis of this dataset can identify which variable is
responsible; the agreement with (14) is only a compatibility check.

### P4 — polar residual is a conditional proxy, not the condition number

In exact arithmetic, and up to eager-bf16 rounding in the recorded value,

$$
r_{\rm pol}^2=
\texttt{muon/polar_residual}^2
=\frac1q\sum_i\left(p(\sigma_i)^2-1\right)^2. \tag{15}
$$

If $k$ hard modes have $p(\sigma_i)\approx0$ and dominate decoherence, then
$r_{\rm pol}\approx\sqrt{k/q}$ and (13) predicts

$$
\epsilon\approx\tau_r r_{\rm pol}. \tag{16}
$$

Thus the hard-mode submodel predicts log-log slope one between role-controlled
<code>muon/replay_update_relerr</code> and
<code>muon/polar_residual</code>, after controlling for factored-scale
dispersion and decay dilution. It also predicts lower
<code>muon/cos_nesterov_final</code> with higher error, because a spread
spectrum is rotated more by polar reweighting.

Equation (16) is deliberately weaker than the Fréchet model. The polynomial
is oscillatory: it can have $p(\sigma)\approx1$ and large $|p'(\sigma)|$.
Consequently two matrices can have the same polar residual and different
condition numbers. Failure of (16) would reject <code>polar_residual</code>
as a proxy, not spectra-dependent amplification.

### Required precision

[I0001](../investigations/0001-seed-variation/conclusion.md) gives a
3.5% sd-relative initialization floor for
<code>muon/replay_update_relerr</code>; this dataset has only one data
ordering. Following its rule, a residual from a quantitative prediction
should be at least 7–10.5% relative (two to three floor units) before being
called a falsification with the existing replication level.

Applied to (14), the two-floor acceptance ranges are approximately

$$
0.861\leq\epsilon_{896}/\epsilon_{768}\leq0.991,
\qquad
0.805\leq\epsilon_{1024}/\epsilon_{768}\leq0.927.
$$

Using three floor units widens them to approximately $[0.829,1.023]$ and
$[0.775,0.957]$. These are heuristic noise-floor bands, not confidence
intervals: the 3.5% estimate is d12-only, while d14 and d16 each have one
seed. The existing deviations from the $q^{-1/2}$ prediction are only 0.9%
and 2.5% relative, below the floor.

For prospective discrimination between exponents $1/2$ and $1/4$, a width
ratio $R$ separates their endpoint predictions by

$$
\Delta(R)=R^{-1/4}-R^{-1/2}.
$$

With independent arms and $\sigma=0.035$, the two-standard-error precision
rule $\Delta\geq2\sigma\sqrt{2/n}$ gives:

| width span $R$ | endpoint separation | $n$ per arm, two-SE rule | $n$ per arm, two-sided $t$, 80% power |
|---:|---:|---:|---:|
| 1.33 | 6.46 pp | 3 | 6 |
| 1.50 | 8.71 pp | 2 | 4 |
| 2.00 | 13.38 pp | 1 | 3 |
| 2.50 | 16.28 pp | 1 | 3 |

The first count treats the 3.5% floor as known and is a precision heuristic,
not literal small-sample power; the final column is the exact equal-variance
two-sample $t$ calculation at two-sided $\alpha=0.05$. A roughly twofold span
therefore makes the exponent contrast cheap. E03's present 1.33-fold span and
six seeds per arm also meet the stated 80% calculation, provided the d12
initialization floor transfers across widths.

Two endpoint ratios cannot separate a smaller exponent from a floor. That
comparison needs curvature across at least three widths, preferably spanning
about twofold; three is an identification minimum, not a power guarantee.
In particular, the fitted linear floor above gives
$f(R)=0.84R^{-1/2}+0.16$. Across a twofold span, the pure power fitted to its
endpoints has exponent 0.407, and at the geometric midpoint the two curves
differ in log space by only 0.00227 (0.23%). A three-width curvature contrast
has standard error $0.035\sqrt{1.5/n}$, so even the two-SE heuristic would
need about 1,400 seeds per width, not six. At E03's current 1.33-fold span the
curvature is smaller still. More widths can improve this, but the current
18-run budget is powered for the $1/2$-versus-$1/4$ endpoint contrast, not for
distinguishing a 16%-of-baseline floor from a freely fitted reduced exponent.

I0001 reports a 7.3% floor for <code>muon/cos_raw_final</code>, not for every
stage cosine; the floors for <code>muon/cos_nesterov_final</code> and
<code>polar_residual</code> must be established before using the P4
associations confirmatorily.

### Instrument dependency

E03 cannot currently run through the official schema-v3 runner. Its manifest
validator permits only <code>depth</code>, <code>seed</code>,
<code>shadow</code>, <code>periodic_points</code>, <code>checkpoints</code>,
<code>deep_schedule</code>, and <code>head_dim</code>; it rejects
<code>aspect_ratio</code>. The runner also does not pass that argument to
<code>base_train.py</code>, and its verifier hard-codes realized width as
$64\,\text{depth}$. P3 therefore depends on a new immutable manifest/runner
path that carries <code>aspect_ratio</code> and verifies the realized model
width. This is in addition to E03's own stated telemetry-v4 dependencies.

## 8. What this model does not explain

- It bounds amplification but does not derive the absolute 3–10% level. That
  needs the magnitude, covariance, and stage location of compiler-dependent
  rounding errors $\Xi_j$.
- It does not predict the observed role ordering from names or shapes. Actual
  post-MuonEq spectra, perturbation orientation, factored scales, and mask
  margins are missing.
- MP is not a model of trained transformer gradients. Its rectangular radial
  calculation happens to give an exponent near one half, while its square and
  isotropic calculations do not explain the attention trend.
- The $q^{-1/2}$ law assumes a bounded number of sensitive modes. It is a
  falsifiable reduction, not a consequence of Marchenko–Pastur theory.
- The smooth perturbation analysis does not cover MuonEq row-norm clamps,
  second-moment clamps, bf16 binade boundaries, or cautious-mask flips.
- Width and depth are structurally unidentifiable in the existing data:
  width is $64\,\text{depth}$ in every run, so no analysis can separate them.
  P3 is a prospective fixed-depth prediction for E03. Schedule, horizon, and
  the absolute 400-step momentum ramp also change along the current size ray,
  and I0003's alignment caveat remains in force.
- It does not cover the exactly zero first updates, where the relative channel
  is undefined and 69.2% of matrices are inactive.
- It does not say that the eager reference is uniquely correct, or that lower
  decoherence improves loss. E05 is the separate causal intervention for that
  question.
- It does not turn <code>polar_residual</code> or the recorded Muon stage
  cosines into production intermediates. Those are eager-reference-frame
  measurements carrying <code>muon/replay_update_relerr</code> as their
  calibration error.

The central claim of this draft is therefore limited but sharp: **five-stage
Polar Express is a high-condition-number spectral polynomial near small and
transition singular modes; compiler-dependent early rounding is multiplied
by known gains, after which two norm-preserving maps erase radial disagreement
but retain or anisotropically reshape angle.** Width and role enter through
the post-MuonEq spectrum, the orientation of rounding error, and the factored
and cautious post-maps. A universal $q^{-1/2}$ decline is a plausible
finite-sensitive-mode prediction for a fixed-depth width experiment. It is
numerically compatible with the current size ratios within their floor, but
those data cannot support a width claim, and their monotone residual is not
explained by the pure reduction. A width-independent floor is permitted by
the general rounding model but remains an additional assumption rather than
a prediction derived from it.
