# T01 — warmdown curvature as a saturated schedule response

Status: draft, round 3

This is a theory note: it proposes a mathematical model and predictions. It is
not evidence that learning-rate warmdown causes sharpening. Round 2 added one
explicitly exploratory same-data shape diagnostic because the default
warmdown itself sweeps the LR; that diagnostic can reject a functional form,
but cannot establish causality. Round 3 records an independent-implementation
cross-check on that same dataset, makes the lack of a saturation mechanism
explicit, and re-ranks the E02 predictions by attribution and power. The
motivating measurement comes from
[I0005](../investigations/0005-certified-curvature-trajectory/conclusion.md);
the proposed intervention is
[E02](../experiments/E02-warmdown-sharpening.md).

## Result in one line

The naive edge-of-stability model is not compatible with T01. It predicts a
20-fold curvature rise over the default warmdown, and even its linear-momentum
version predicts a continuing rise; the measured direction-normalized
curvature rises about 4.8-fold and has already plateaued by progress 0.65,
when the learning-rate multiplier is still about 0.56.

The surviving candidate functional model is a **putative LR-gated,
empirically capped response**. If $R$ is certified curvature along the
probe-gradient direction, $r$ is the LR multiplier, and $b_i$ is a run offset,
then

\[
\log R_i(r)=b_i+a\min\{-\log r,z_s\},
\qquad a>0.
\tag{1}
\]

The first draft calibrated $a=2.70$ and $z_s=0.577$ from two phase summaries.
That calibration was too crude. A round-2 fit to the 70 existing certified
warmdown points gives, exploratorily,

\[
a\approx3.04,
\qquad z_s\approx0.374,
\qquad r_s=e^{-z_s}\approx0.688,
\qquad e^{a z_s}\approx3.12.
\tag{2}
\]

Thus $\log R$ has slope $-a\approx-3.04$ against $\log r$ early in warmdown,
slope zero late, and a bend at $\log r\approx-0.374$. A straight power law is
not another interpretation of the same curve: it is a different functional
claim, and the existing data already fit it much worse. Equation (1) remains a
constitutive model, not a microscopic derivation from Muon. In particular,
$r_s=0.688$ is a fitted bend location, not a derived optimizer threshold.

## 1. Phenomenon and measurement

I0005 reports the following at d12 on the `shadow_fp32` acceptance arm,
restricted to checkpoints where the gradient-direction verdict passes:

- `curvature/vhv_gradient` is flat under constant learning rate, rises after
  warmdown begins at normalized progress $p_0=0.350$, and plateaus over the
  last third of training, $p\ge 0.65$. Its reported warmdown/pre-warmdown
  fold is $4.76$ (seed range $4.45$–$7.12$).
- `curvature/gg`, the squared norm of that probe gradient, rises $2.73$-fold.
- `curvature/gHg` rises $15.6$-fold. Within runs, roughly 61% of its log rise
  is attributed to `vhv_gradient` and 37% to `gg`.
- All five seeds rise during warmdown; none has an agreed pre-warmdown trend.

The 4.76-fold number is a ratio of the median over *all warmdown checkpoints*
to a pre-warmdown median. It is not an endpoint/onset ratio. Combining it with
the endpoint LR ratio of 20 gives the rough number
$\log(4.76)/\log(20)=0.521$, but that quotient is not a matched estimator of a
power-law exponent. On the common checkpoint medians, the endpoint/onset
`vhv_gradient` ratio is about $10.48/3.60=2.91$, and a regression over all
warmdown points is the appropriate shape test.

These are not three independent curvature observations. With probe loss
$L_p$, gradient $g=\nabla L_p$, Hessian $H=\nabla^2 L_p$, and
$v=g/\lVert g\rVert$,

\[
R := v^\top H v
=\frac{g^\top H g}{g^\top g},
\qquad
\texttt{gHg}=R\,\texttt{gg},
\qquad
\texttt{eta_star}=R^{-1}.
\tag{3}
\]

The model below targets $R$, the direction-normalized 61% component. It
does not model the gradient-norm growth. The separately summarized medians in
I0005 need not multiply exactly, which is why $4.76\times2.73\ne15.6$.

The scope is narrow. $L_p$ is the loss on one 256-token sequence evaluated
on an IEEE-fp32 shadow copy. It is not the 2048-token training loss, the native
bf16 surface, or an expectation over data. $R$ is a Rayleigh quotient along
one direction; it is neither $\lambda_{\max}(H)$ nor certified curvature
along the applied update.

## 2. The actual step made by Muon

Let $p\in[0,1]$ be normalized training progress, $p_0$ the warmdown onset,
and $f$ `final_lr_frac`. Ignoring step rounding, the schedule in
[`base_train.py`](../../nanochat/scripts/base_train.py) is

\[
r(p;p_0,f)=
\begin{cases}
1, & p\le p_0,\\[3pt]
1-(1-f)\dfrac{p-p_0}{1-p_0}, & p>p_0.
\end{cases}
\tag{4}
\]

For a Muon matrix gradient $G_t$, the relevant parts of the implemented
update in [`optim.py`](../../nanochat/nanochat/optim.py) are

\[
\begin{aligned}
M_t &= \beta_t M_{t-1}+(1-\beta_t)G_t,\\
N_t &= (1-\beta_t)G_t+\beta_t M_t,\\
U_t &= \mathcal N\!\left(\mathcal P(N_t),V_t\right),\\
W_{t+1} &= W_t-\eta_W r_t
  \sqrt{\max(1,m/n)}
  \left(U_t+w_t W_t\odot C_t\right).
\end{aligned}
\tag{5}
\]

Here $\mathcal P$ includes row equilibration and five Polar Express steps;
$\mathcal N$ includes the Frobenius-norm snap and factored second-moment
scaling; and $C_t$ is the cautious-decay mask. For a nondegenerate Nesterov
input and $W\in\mathbb R^{m\times n}$, provided neither protective
denominator clamp is active, the ideal reference data component obeys, in
exact real arithmetic,

\[
\lVert U_t\rVert_F=\sqrt{\min(m,n)},
\qquad
\lVert\Delta W_{t,\mathrm{data}}\rVert_F
=\eta_W r_t\sqrt m.
\tag{6}
\]

The second equality includes the shape adjustment in `_compute_muon`. If `lr`
means the scalar passed into `muon_step_fused` after that adjustment, the same
statement is `lr * sqrt(min(m,n))`; if it means `group["lr"]`, as it usually
does in the recipe, the answer is `group_lr * sqrt(m)`. The factored scaling
preserves the snapped norm algebraically because `v_norm_new` is the norm
after multiplication by `step_size`, and `final_scale` multiplies it back by
`v_norm / v_norm_new`.

This makes the Muon data component **norm-controlled algebraically**, but not
literally exact as an applied production update:

- if the Nesterov combination is zero, the output remains zero rather than
  being snapped to $\sqrt{\min(m,n)}$;
- bf16 multiplications round the snap and norm-preserving rescale;
- cautious decay adds a direction-dependent component;
- nonmatrix parameter groups use AdamW; and
- the compiled update differs from the eager reference by 3–10% per matrix.

As a scale check, recorded nondegenerate eager-reference norm deviations at
d12 have maximum relative error about $2.9\times10^{-7}$: tiny, but not
bit-exact. E02 records the actual full-model `update/direction_norm`, so a
serious test should use both $r_t$ and the recorded ratio

\[
r_{\delta,t}
=\frac{\lVert\delta_t\rVert}
       {\operatorname{median}_{\mathrm{pre}}\lVert\delta\rVert}.
\tag{7}
\]

The Muon momentum coefficient is also scheduled. It falls linearly from 0.97
to 0.90 over exactly the same interval as (4), including in E02's constant-LR
arm. In the ideal nondegenerate data component, $\beta$ does not appear in
(6): momentum changes the direction presented to the polar map, not its final
Frobenius norm. It can still change the cautious-decay mask and hence the full
step norm.

There is also no steady-state $1/(1-\beta)$ gain in nanochat's momentum
parameterization. The code uses
$M_t=\beta M_{t-1}+(1-\beta)G_t$, whose DC gain is one; for a constant gradient,
$M_t\to G$ and $N_t\to G$. The familiar $1/(1-\beta)$ factor belongs to the
unnormalized recurrence $M_t=\beta M_{t-1}+G_t$. Thus the LR exponent is not
secretly carrying a 3.3-fold momentum *scale* change. The remaining confound
is directional and causal, not multiplicative step size.

## 3. Why edge-of-stability is not the model

### 3.1 The valid local quadratic inequality

For an arbitrary unit update direction $u$, step length $s>0$, and local
step $\delta=-su$, the quadratic loss change is

\[
L(\theta+\delta)-L(\theta)
=-s g^\top u+\frac{s^2}{2}u^\top H u.
\tag{8}
\]

If $g^\top u>0$, non-increase of this quadratic model requires

\[
\kappa_u:=u^\top H u
\le \frac{2g^\top u}{s}
=\frac{2\lVert g\rVert\cos\phi}{s}.
\tag{9}
\]

For gradient descent, $u=g/\lVert g\rVert$ and
$s=\eta\lVert g\rVert$, so (9) reduces to the familiar
$R\le2/\eta$. For a norm-controlled step, $s$ does not contain
$\lVert g\rVert$; the right side also depends on gradient norm and alignment.
Even this is only a one-step descent condition, not a theorem that a training
trajectory must track the boundary.

The probe version of (9) can be written in recorded quantities as

\[
\kappa_{\delta}\le
\frac{-2\,\texttt{update/p1}}
     {\texttt{update/direction_norm}^2}.
\tag{10}
\]

But (10) does not rescue an edge interpretation: the applied update is made
from a different training batch, its probe directional curvature never
certifies, and the measured $R$ is along $g$, not along $\delta$.
Consequently neither $\eta R$ nor $\eta\lambda_{\max}$ is a valid stability
statistic here.

### 3.2 The numbers reject the direct inverse-step law

The default schedule takes $r$ from 1 to 0.05. A direct ceiling
$R\propto1/r$ predicts a 20-fold endpoint/onset rise. The 4.76-fold summary is
not the matched comparison because it divides a whole-warmdown median by a
pre-warmdown median. On the common median trajectory, the matched
endpoint/onset rise is only 2.91-fold. The inverse law therefore overpredicts
it by a factor of 6.87, well beyond the 50–75% practical curvature detection
threshold established by I0001.

The trajectory is an even stronger contradiction. At $p=0.65$, where I0005
reports the start of the plateau, (4) gives

\[
r(0.65;0.35,0.05)=0.5615.
\tag{11}
\]

An inverse-step law has gained only $1/0.5615=1.78$ there and must continue
growing by another factor 11.2 as the LR falls to 0.05. The observed response
has instead largely stopped. An uncapped law cannot fit both facts.

Using matched common endpoints, `gg` grows 2.52-fold, so the probe-gradient
norm grows $\sqrt{2.52}=1.59$. If one nevertheless inserted that norm into
(9), held alignment fixed, identified update-direction curvature with
gradient-direction curvature, and treated the probe as the training loss, the
putative boundary would grow roughly $20\times1.59=31.7$-fold. (Using the
published phase summaries gives the question's rough 33-fold figure.) The
matched $R$ rise is only 2.91-fold, so this **fixed-alignment, fixed-transfer
ceiling-tracking model is ruled out**: the trajectory moves farther below that
constructed boundary.

That conclusion does not mathematically eliminate every model called a
“stability ceiling.” Equation (9) concerns $\kappa_u$ along the step, while the
measurement is $R$ along the probe gradient. Its cosine is the probe-gradient
versus full applied-update cosine, not `muon/cos_raw_final`. It is already
recoverable from recorded quantities:

\[
\cos\phi_p=
\frac{-\texttt{update/p1}}
{\sqrt{\texttt{curvature/gg}}\,
 \texttt{update/direction_norm}}.
\tag{11a}
\]

On the existing warmdown checkpoints this cosine is of order $10^{-5}$ and
changes sign late in training; the update is nearly orthogonal to, and
sometimes ascent for, this one probe. The training batch is different, so that
is not an optimizer failure. It means the probe cannot support a productive
step ceiling interpretation without an additional relationship between
$\kappa_u$ and $R$.

The saturation model in (1) is not a ceiling-tracking model: its cap is a cap
on this measured Rayleigh quotient, not the quadratic productivity boundary in
(9). D3 therefore does not refute it. It does remove “edge of stability” as a
proposed microscopic justification for the cap.

### 3.3 Momentum makes a closer number, but still the wrong model

There is one tempting numerical coincidence. Remove every nonlinear Muon
operation after $N_t$ and apply $N_t$ directly to a scalar quadratic
$L(x)=\lambda x^2/2$. The exact recurrence is

\[
\begin{bmatrix}x_{t+1}\\M_t\end{bmatrix}
=
\begin{bmatrix}
1-\eta\lambda(1-\beta^2)&-\eta\beta^2\\
(1-\beta)\lambda&\beta
\end{bmatrix}
\begin{bmatrix}x_t\\M_{t-1}\end{bmatrix}.
\tag{12}
\]

The Jury conditions for both roots to lie inside the unit disk give

\[
0<\eta\lambda<
C(\beta):=
\frac{2(1+\beta)}{(1-\beta)(1+2\beta)}.
\tag{13}
\]

Here $C(0.97)=44.7$ and $C(0.90)=13.6$. Combining the momentum and LR
endpoints predicts a ceiling ratio

\[
\frac{C(0.90)/0.05}{C(0.97)/1}=6.08,
\qquad
\frac{\log6.08}{\log20}=0.602.
\tag{14}
\]

This is closer to 4.76 than 20 is, but it does not explain the trajectory. At
$p=0.65$, $\beta=0.9377$ and the corresponding boundary ratio is only

\[
\frac{C(0.9377)/0.5615}{C(0.97)}\approx0.86,
\tag{15}
\]

while the observed curvature has already risen and plateaued.

More importantly, (12) is not Muon. In the ideal scalar limit the polar map
and norm snap turn $N_t$ into its sign, so

\[
x_{t+1}=x_t-\eta\,\operatorname{sign}(N_t).
\tag{16}
\]

Positive rescaling by $\lambda$ disappears from the normalized update. There
is no $\eta\lambda$ linear-stability boundary at all in this scalar model.
In matrices, curvature can affect directions and singular subspaces, but a
new analysis is required; the linear-momentum boundary cannot be imported
through the polar map. Equations (12)–(15) are therefore a rejected surrogate,
not a derivation for nanochat.

## 4. Working model and the free within-run test

Define

\[
z(p):=-\log r(p),
\qquad y_i(p):=\log R_i(p).
\tag{17}
\]

The model assumes that fractional step shrinkage is the causal clock, that
log curvature has constant susceptibility $a$ before a finite cap, and that
relaxation is faster than the 0.05 progress grid. Its state equation is

\[
\frac{dy_i}{dz}=
\begin{cases}
a,&z<z_s,\\
0,&z>z_s,
\end{cases}
\qquad y_i(0)=b_i,
\tag{18}
\]

which integrates to (1). The run intercept $b_i$ absorbs a curvature-level
offset. The model does not derive $a$ or $z_s$ from architecture.

Equation (18) says exactly what is assumed to saturate: the **onset-relative
probe-gradient Rayleigh quotient**,

\[
\frac{R_i(r)}{R_i(1)}
=\exp\!\left(a\min\{-\log r,z_s\}\right),
\tag{18a}
\]

which reaches the fitted fold $e^{a z_s}\approx3.12$. It does not say that a
Hessian eigenvalue, the Hessian spectrum, the training-loss curvature, or an
optimizer stability margin reaches a capacity. Because $b_i$ is free, it does
not even posit a common absolute cap across runs. A plateau in $R$ could come
from eigenvalues ceasing to grow, from the probe-gradient direction ceasing to
rotate toward sharper eigenspaces, or from both.

There is **no mechanism yet for $r_s\approx0.688$**. Both $r_s$ and the hard
cap were inferred from the same 70 points whose plateau motivated the hinge
family. Calling $r_s$ a threshold is therefore shorthand for a fitted bend,
not a derivation of why Muon or this transformer should change regime after a
31.2% LR reduction.

A mechanism would need an independently measured state $q_t$ and a dynamical
law that predicts its capacity $q_s$ before fitting the curvature bend; only
then could $r_s$ be derived from the crossing $q_t=q_s$. A concrete diagnostic
would densely measure, around the bend, a certified Hessian spectral or
projected-Hessian decomposition on the same probe together with the
probe-gradient projection weights, the actual update norm $r_{\delta,t}$, and
a stochastic-gradient noise/step statistic. That would distinguish an
eigenvalue cap from saturation of the direction weights and test candidate
drivers such as a step-to-noise transition. Re-locating the plateau more
precisely in $r$ alone would not supply a mechanism.

### 4.1 Sharp prediction in log-LR coordinates

Let $x=\log r=-z$. Then the local slope is

\[
\frac{d\log R}{d\log r}=
\begin{cases}
-a\approx-3.04,& -0.374<x\le0,\\
0,& x<-0.374.
\end{cases}
\tag{19}
\]

The bend is at

\[
\log r_s\approx-0.374,
\qquad r_s\approx0.688.
\tag{20}
\]

So the prediction is a continuous **broken straight line**, not a globally
straight line and not merely unspecified curvature. In time order, the early
warmdown segment rises steeply and the late segment is horizontal. The
onset-relative fold at the cap is about 3.12. For orientation:

| LR multiplier $r$ | 1.00 | 0.90 | 0.80 | 0.70 | 0.688 and below |
|---|---:|---:|---:|---:|---:|
| predicted $R/R(r=1)$ | 1.00 | 1.38 | 1.97 | 2.96 | 3.12 |

An uncapped power law predicts one slope, $-\alpha$, from $r=1$ through
$r=0.05$. In particular, $\alpha=0.521$ predicts substantial growth after
$r=0.688$ rather than a plateau.

### 4.2 Round-2 same-data diagnostic

The existing dataset contains 14 certified warmdown points per d12 run, 70
points total. Checkpoints are not independent replicates, so the five runs are
the replication units. With a run fixed effect, compare

\[
\begin{aligned}
M_P:&\quad y_{ij}=b_i+\alpha z_{ij}+\epsilon_{ij},\\
M_S:&\quad y_{ij}=b_i+a\min(z_{ij},z_s)+\epsilon_{ij},\\
M_2:&\quad y_{ij}=b_i+a_1\min(z_{ij},z_s)
             +a_2\max(z_{ij}-z_s,0)+\epsilon_{ij}.
\end{aligned}
\tag{21}
\]

A read-only exploratory fit to the already-extracted certified table gives:

| model | pooled shape estimate | pooled SSE | leave-one-seed-out centered SSE |
|---|---|---:|---:|
| straight power $M_P$ | $\alpha=0.277$ | 9.425 | 9.647 |
| saturated hinge $M_S$ | $a=3.038$, $z_s=0.374$ | 3.202 | 4.122 |
| free two-slope $M_2$ | $a_1=3.033$, $a_2=0.021$, $z_s=0.368$ | 3.190 | 4.105 |

The near-zero fitted late slope is the relevant result. The saturated model's
predictive error is less than half the straight model's even when the held-out
run contributes only an intercept. Four of five held-out runs favor the hinge;
`d12-s7` is the exception. Per-run saturated break estimates range from
$z_s=0.238$ to $0.576$, so the common bend is not precise enough to call
universal.

After this fit was made, the coordinator independently regressed the same
certified warmdown channel on log LR without seeing the round-2 result. That
implementation also obtained pooled $\alpha=0.277$ (agreement to three
digits), found a mean quadratic term of $-0.319$ that was negative in all five
runs, and found the local slope collapse from about 1.9 early to about zero
late; see [X01](X01-warmdown-exponent.md). This is a
genuine implementation cross-check: two different analyses recover the same
pooled exponent and both reject a globally straight power law. It is not
independent-data confirmation, because both use the same five runs.

This check rejects a globally straight power law as a description of the
existing trajectory. It does **not** confirm LR causality: the hinge family was
suggested by the same reported plateau, and LR, progress and momentum direction
remain confounded. It also corrects the first-draft calibration: using a phase
ratio as an asymptotic cap put the bend too late.

## 5. Quantitative predictions for E02

For an arm with onset $p_0$ and final fraction $f<r_s$, solving
$r(p_s)=r_s$ gives

\[
p_s(p_0,f)
=p_0+(1-p_0)\frac{1-r_s}{1-f}.
\tag{22}
\]

Using the exploratory $r_s=0.688$, the five E02 arms give:

| E02 arm | $p_0$ | $f$ | predicted rise onset | predicted plateau onset | onset-relative cap fold |
|---|---:|---:|---:|---:|---:|
| A reference | 0.350 | 0.05 | 0.350 | **0.563** | **3.12** |
| B early | 0.250 | 0.05 | 0.250 | **0.496** | **3.12** |
| C late | 0.650 | 0.05 | 0.650 | **0.765** | **3.12** |
| D constant LR | none | 1.00 | none | none | **1.00** |
| E shallow decay | 0.350 | 0.30 | 0.350 | **0.640** | **3.12** |

These predictions concern certified `curvature/vhv_gradient`, not
$\lambda_{\max}$. They are not equally attributable or equally powered. For
this LR-gated model, {A, D, E} is the single-factor LR contrast because its
progress and momentum trajectories are identical; {A, B, C} is not. Their
evidential order is:

1. **D is decisive.** A and D have the same `warmdown_ratio`, normalized
   progress, checkpoint grid, and bit-identical Muon momentum trajectory; only
   the LR decay differs. Equation (1) predicts an onset-relative fold of 3.12
   in A and 1.00 in D. The 212% separation is about
   $2.12/0.29=7.3$ times the conservative 29% seed-relative reference floor on
   a crude linear scale. That is not a formal seven-sigma test—the reference
   floor is for a level, whereas this is a within-run fold—but it correctly
   identifies D as the largest and cleanest contrast. D should show no hinge
   even though its momentum changes from 0.97 to 0.90.
2. **E is a clean but weaker LR-only contrast.** A and E also have identical
   progress and momentum trajectories. The model predicts the same cap because
   both final multipliers lie below $r_s$, whereas a globally uncapped
   $r^{-0.521}$ law predicts only $0.30^{-0.521}=1.87$-fold in E. Its exact
   bend-location prediction is weak: $p_s=0.640$ versus 0.563 in A is a
   separation of 0.077, only about 1.5 deep-checkpoint spacings. The existing
   per-run $z_s$ range maps to $p_s\in[0.495,0.650]$, a span of 0.155. A range
   is not a standard deviation and therefore does not by itself prove a power
   number, but with three seeds per arm the A/E bend displacement should be
   treated as descriptive, not a sharp test.
3. **A/B/C test a shared schedule clock, not LR attribution.** In
   `base_train.py`, both `get_lr_multiplier` and `get_muon_momentum` recompute
   their onset and duration from the same `warmdown_ratio`. Moving that
   argument moves LR and momentum together. B and C have substantial timing
   leverage against a fixed-progress account, but a bend that follows them
   could be LR-gated, momentum-gated, or a response to both. Their exact
   plateau locations are therefore useful model descriptions, not clean
   causal tests of equation (1)'s LR clock.

A, B, C and E should still collapse onto (1) against the realized LR
multiplier if the constitutive law transports. Against
`update/direction_norm`, the bend should occur at a comparable fractional
data-step shrinkage, subject to AdamW and decay contamination. In capped arms,
`curvature/eta_star` has the reciprocal onset-relative fold, about
$1/3.12=0.321$; this is not independent evidence. At matched checkpoints,
`curvature/gHg / curvature/gg` follows (1) identically, while the model makes
no prediction for `gg` or `gHg` alone.

The 3.12 number uses the onset value, whereas E02's drafted fold uses a
pre-phase median and eight tail points. Those estimands must not be compared as
if they were identical; the model predicts equality of the A and E cap under
either consistently applied definition.

The hinge family was suggested by the reference trajectory's plateau and then
fit to that same trajectory, so its apparent fit advantage remains
exploratory. D and E escape that same-data circularity only prospectively: if
their predictions are frozen before their data are seen, they are new
single-factor interventions. D tests whether LR shrinkage is necessary; E
tests transport of the fitted cap to a shallower decay. Neither retroactively
turns the round-2 calibration into confirmatory evidence, and E still inherits
the same-data estimates of $a$ and $r_s$.

## 6. Falsification and required precision

All curvature tests must use `acceptance_arm == "shadow_fp32"`, defined rows,
and `curvature/verdict_code_gradient == 0`. Tests on native curvature, random
directions, or update directions do not test this model because those
directions are uncertified.

The applicable initialization floors from
[I0001](../investigations/0001-seed-variation/conclusion.md) are 25%
sd-relative for `eta_star` and therefore the same underlying $R$ channel,
and 29% for `gHg`. The practical rule is that a between-run curvature-level
effect must be about 50–75% before five d12 runs can separate it from
initialization noise. The floor covers one shared data order only.

The model should therefore be tested with within-run folds and times, not
absolute curvature levels. I0005 measured a 23% across-seed spread for the
within-run `vhv_gradient` fold. Against those floors, the tests rank as
follows:

1. **Constant-LR D is the decisive falsifier.** A reference-like D trajectory
   means a break at $0.350\pm0.05$ followed by an onset-relative fold near
   3.12 despite $r\equiv1$. That directly falsifies LR/step shrinkage as the
   clock in (18), including its D-arm cap prediction of 1.00. More
   conservatively, an identified D break plus a consistent fold above the 75%
   practical curvature threshold is a falsifier; smaller changes are
   unresolved, not confirmation. If D sharpens, what remains is the empirical
   saturation-shaped description of the original reference runs and the
   rejection of a globally straight power law. Progress-gated and
   momentum-direction-gated saturation both remain live. D alone cannot choose
   between them because progress 0.350 and the momentum onset are still
   synchronized in that arm.
2. **Functional shape is already testable.** From the fitted bend to the end,
   an uncapped $\alpha=0.521$ power law predicts another
   $(0.688/0.0506)^{0.521}\approx3.90$-fold rise. Saturation predicts 1.00.
   This 290% separation is far above the 23% fold spread. The free diagnostic
   in §4.2 therefore has adequate shape leverage even though the 70 checkpoint
   rows are not 70 independent samples.
3. **Shallow-decay E tests cap transport.** The model predicts the A-versus-E
   log-cap contrast to be zero; the rough uncapped $0.521$ law predicts a
   substantial contrast. E02's precommitted interval on consistently defined
   within-run folds is the test. Equality of noisy absolute levels is not
   evidence.
4. **A/B/C test shared-scheduler lock.** Their onset span of 0.400 gives useful
   power to distinguish a moving shared-schedule response from a fixed
   progress-0.350 response. It gives no power to distinguish LR from momentum,
   because both schedules move together.
5. **Exact bend locations are descriptive.** In particular, A versus E differs
   by only 0.077 while historical per-run estimates span 0.155. Agreement with
   (22) to one 0.05 checkpoint is too strong a falsification rule at three
   seeds. A large qualitative failure to bend, or continued late growth, is
   informative; a one-grid miss is not.
6. **Slope replication unit.** Estimate straight, early and late slopes per
   run, then report their five-run distribution or use a run-block bootstrap.
   Treating checkpoints as independent would manufacture precision. A useful
   numerical falsifier for the revised hinge is a common late-slope interval
   excluding zero or a common early-slope interval excluding 3.04 on new data.

The five existing runs reject the straight functional form descriptively and
calibrate (1); they cannot establish which synchronized clock caused the
hinge. That requires an intervention such as E02.

## 7. Candidate models, ranked

Two quantitative causal clocks remain compatible with the existing co-timing
and are ranked below by how much of the known shape they encode. A momentum
clock is also live, but does not yet have a mathematical transfer law.

1. **LR/step-gated saturated response, equation (1).** This is the working
   model. It accounts for flatness, onset and early saturation, and it wins the
   free shape comparison. Its clean predictions are no D-arm hinge and the
   same cap in E as A; the arm-specific plateau times in (22) are exploratory
   calibration targets.
2. **Progress-gated saturated response.** Replace $z=-\log r$ in (18) by a
   function of $(p-0.35)_+$, calibrated to the same existing curve. It fits
   the existing trajectory identically because $r$ is a deterministic
   one-to-one function of progress during this one warmdown. It predicts the
   same onset and bend progress in every E02 arm, including D. D cleanly
   distinguishes it from rank 1. B and C can distinguish fixed progress from a
   moving shared-scheduler clock, but cannot say whether that clock is LR or
   momentum.

Two useful benchmark families do not survive the known shape:

- **Uncapped power response:** $R\propto r^{-\alpha}$. Even after refitting
  $\alpha$ freely, its pooled and held-out errors are more than twice the
  saturated hinge's. The specific $\alpha=0.521$ number also mixes a phase
  ratio with an endpoint LR ratio. This family is descriptively rejected on
  the existing d12 trajectory.
- **Edge-of-stability:** the GD version predicts 20-fold. The exact
  pre-polar momentum surrogate predicts 6.08-fold at the endpoint but a
  boundary ratio of 0.86 at the observed plateau onset. It also predicts a
  different quantity for a different optimizer. Neither is an explanation of
  T01.

A momentum-direction model is a live verbal possibility, not yet a ranked
mathematical model: after polar normalization, $\beta$ can rotate the update
without a known scalar transfer law from that rotation to $R$. Arm D can
show that such a model is needed, but a magnitude law should not be invented
before then.

No further regression on the default runs can choose among LR, progress and
momentum direction as causal clocks: over this trajectory they are
deterministically collinear. The minimal causal intervention is E02 arm D,
which keeps LR constant while momentum changes. A complete LR-versus-momentum
separation additionally needs the converse arm—LR decaying while momentum is
held fixed—which the present recipe cannot express without a schedule change.

## 8. The gradient/update direction gap

Let $v=g_p/\lVert g_p\rVert$ be the certified probe-gradient direction and
$u$ a unit update direction on the same parameter space. Write

\[
u=cv+\sqrt{1-c^2}\,w,
\qquad c=v^\top u,
\qquad w^\top v=0.
\tag{23}
\]

Then

\[
u^\top Hu
=c^2v^\top Hv
+2c\sqrt{1-c^2}\,v^\top Hw
+(1-c^2)w^\top Hw.
\tag{24}
\]

Knowing $c$ and the certified value $v^\top Hv$ does not determine or usefully
bound the other two terms. For $|c|<1$, even if $H$ were positive semidefinite,
$u^\top Hu$ can range from zero to arbitrarily large values unless a spectral
bound is supplied. If one knew $\lVert H\rVert_2\le M$, the generic angle bound

\[
|u^\top Hu-v^\top Hv|
\le2M\lVert u-v\rVert
=2M\sqrt{2(1-c)}
\tag{25}
\]

would follow, but $M$ is precisely the unmeasured spectral quantity.

`muon/cos_raw_final` is not the $c$ in (23). It is a **per-matrix** cosine
between the current logical-training-batch raw gradient and the eager-reference
Muon data direction. The certified $v$ instead comes from one short probe, and
$u$ in (9) is the full actual update including AdamW and decay. Per-matrix
cosines also cannot be averaged into a full-model cosine without norm weights.
The required probe/full-update angle is already given by (11a).

The recorded Muon cosine is nevertheless a useful directional-confound
diagnostic. An exploratory round-2 summary finds that its per-checkpoint median
across matrices rises from about 0.14 before warmdown to about 0.28 in the tail;
the median per-matrix tail/pre ratio is about 1.99, with the same direction in
all five seeds. That is consistent with lower momentum making the update more
raw-gradient-aligned. It cannot *numerically manufacture* `vhv_gradient`, which
is computed independently from $g_p$ and $H_p$ at the current state. It can
change the states subsequently visited and therefore remains a plausible
causal confound for the trajectory.

The minimal measurement that would connect the directions is a certified
two-dimensional projected Hessian on the **same probe loss**:

\[
B=
\begin{bmatrix}
v^\top Hv & v^\top Hu\\
u^\top Hv & u^\top Hu
\end{bmatrix},
\tag{26}
\]

together with the global probe/update cosine. The cosine is already recorded
through `p1`; the cross term and a certified $u^\top Hu$ are not. This can be
attempted offline on saved checkpoints, preferably with the longer/larger
probe selected by E09/E06. If the update direction still fails certification,
the honest result is that no stability claim is available.

## 9. Probe sequence length

The HVP loss uses a mean over valid targets, so increasing sequence length does
not mechanically multiply $R$ by $T$. A minimal sampling model writes the
length-$T$ probe operator as

\[
g_T=\mu+T^{-1/2}\xi_T,
\qquad
H_T=\bar H+T^{-1/2}\Xi_T.
\tag{27}
\]

If $\mu\ne0$, token contributions mix sufficiently, and the denominator stays
away from zero, a delta-method expansion gives

\[
R_T=\frac{g_T^\top H_Tg_T}{g_T^\top g_T}
=R_\infty+O_p(T^{-1/2}),
\tag{28}
\]

with expected bias ordinarily $O(T^{-1})$. Under this model, the warmdown
shape parameters—power exponent, or hinge slopes and bend—are invariant to
leading order; only their sampling variance falls. Moving from 256 to 2048
tokens should reduce a token-sampling standard deviation by about
$\sqrt{8}=2.83$.

This is not a secure prediction for nanochat. Tokens are dependent, document
packing changes target composition, the gradient mean may be small, and the
attention architecture changes regime across $T=512$. As
[E09](../experiments/E09-sequence-length-geometry.md) notes, d12's short-window
layers first become windowed between $T=512$ and $T=1024$. A systematic jump
on that rung would refute (27) as an adequate probe model and show that the
measured hinge parameters are properties of a particular probe operator.

Neither edge-of-stability nor the saturated constitutive law uniquely predicts
a sequence-length exponent. The sharp E09 test for this note is instead
whether the within-state sharpening ratio and, where four anchors allow it,
the fitted warmdown shape are stable across $T$. E09's four saved interior/final
anchors can test a shape ratio but cannot re-estimate the 14-point bend. If the
shape varies materially with $T$, no claim should identify T01 with the
training-loss geometry until a training-length, multirow probe is used.

## 10. Where the working model breaks

- **It is phenomenological.** Constant log susceptibility and a hard cap on
  the onset-relative probe-gradient Rayleigh quotient are assumed. No
  transformer or optimizer calculation derives $a\approx3.04$ or
  $r_s\approx0.688$, and no microscopic capacity has been identified. A smooth
  relaxation model could fit a rounded transition.
- **Causality is unestablished.** The only data have LR onset, momentum onset,
  and progress 0.35 at the same checkpoint. A/B/C move LR and momentum
  together; D and E provide the clean LR interventions within E02.
- **The LR is not the whole applied step.** AdamW parameter groups, cautious
  decay, factored redistribution, bf16 representability, and 3–10% compiled
  Muon replay error all violate the ideal norm calculation. This is why the
  prediction includes `update/direction_norm` as a second x-axis.
- **Momentum can change geometry without changing norm.** The fall from 0.97
  to 0.90 changes the temporal mixture entering Polar Express and can rotate
  the applied update. Equation (1) omits that state.
- **The measured function is not the optimized function.** The model concerns
  a single 256-token probe on an fp32 shadow surface. The training batch is
  different, and native bf16 curvature certifies nowhere.
- **The measured direction can rotate.** A plateau in
  $g^\top Hg/g^\top g$ may reflect changing spectral weights of $g$, not a
  cap on any Hessian eigenvalue. The model intentionally does not name the cap
  a landscape-wide curvature ceiling.
- **The calibration is same-data and coarse.** The bend is selected from 14
  warmdown points per run after the plateau was already known. Missing
  certified points and the 0.05 deep grid limit time resolution; the per-run
  bend range is wide.
- **The noise reference is narrow.** Five seeds vary initialization only; one
  data order and one frozen probe exist. The precision statements do not
  transfer to interventions that alter batching or probe selection.

## 11. What this model does not explain

The note does not explain the 2.73-fold growth of `gg`, and therefore does not
predict the 15.6-fold `gHg` rise. It does not predict $\lambda_{\max}$, the
Hessian spectrum, uncertified update-direction curvature, native-bf16
curvature, loss, generalization, or cross-depth behavior. It does not say why
the cap has its measured value, whether sharpening helps training, or whether
one run is sharper than another. Most importantly, it is not evidence that the
trajectory sits at an edge of stability; the statistic needed for that claim
is not certified or even defined by $\eta\lambda_{\max}$ for this optimizer.
