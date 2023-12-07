# Mockup for SwE Benchmark

Mockup of the sandwich estimator (SwE) calculation using simulated data for benchmarking purposes.

## Quick Start

[Install Rust](https://www.rust-lang.org/tools/install). On Linux this is a one-liner:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

> Hint: Later on you can update your Rust toolchain with [`rustup update`](https://rust-lang.github.io/rustup/basics.html).

Clone this repository and invoke [cargo](https://doc.rust-lang.org/cargo/), Rust's build tool.

```bash
git clone --recurse-submodules https://github.com/benkay86/swe-mockup.git
cd swe-mockup
cargo run --release
```

> Hint: Pull the latest changes to your local git repository with `git pull --recurse-submodules`.

> Hint: Use the `--release` profile to enable compiler performance enhancements.

Sample benchmark output:
```
Benchmark of multiple, parallel SwE computations.
File mock-data.npz not found.
Consider running mock-npz to generate data.
Generating mock data on the fly... done.
Mock data parameters:
Number of observations: 8192
Number of features: 55278
Number of predictors: 8
Number of blocks: 1800
Number of parallel repetitions: 20
Thread pool has 30 cpus.
Computing SwE...  done.
Time elapsed: 48.608350496s
That's 2.430417524s per repetition.
```

## Generating Mock Data

By default, the benchmarks will randomly generate a new set of mock data on each run. If you want the most consistent comparisons possible between benchmarks and runs, generate a single mock data set in advance and save it to disk. The data will be saved in the [NumPy format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format) to `mock-data.npz` by default. The benchmarks will automatically use data from the `mock-data.npz` file if it is present (instead of generating new data).

```bash
cargo run --release --bin mock-npz # generate data
cargo run --release                # benchmark using generated data
```

To use the mock data in Matlab, use the [npy-matlab package](https://github.com/kwikteam/npy-matlab). This package does not support `*.npz` files, so first you will have to unzip the data and rename the data structures. Then you can call `readNPY()` in Matlab to load the files.

```bash
mkdir -p mock-data
unzip mock-data.npz -d mock-data
for FILE in mock-data/*; do mv ${FILE} ${FILE}.npy; done
```

## Selecting a Benchmark

The available benchmarks are listed under [src/bin](./src/bin). The default benchmark is a parallel computation of multiple sandwich estimator covariance matrices. To select a different benchmark, sich as a single computation of the SwE, specify the desired benchmark with `--bin`. For example:

```bash
cargo run --release --bin benchmark-single
```

## Matlab Benchmarks

To run the Matlab benchmarks, first generate some mock data and prepare it as above:

```bash
# Generate mock data
cargo run --release --bin mock-npz
# Prepare the data for Matlab
mkdir -p mock-data
unzip mock-data.npz -d mock-data
for FILE in mock-data/*; do mv ${FILE} ${FILE}.npy; done
```

Then enter the [`matlab`](./matlab) directory and run the desired benchmark.

```bash
cd matlab
matlab -nodesktop -r benchmark_multi
```

## Troubleshooting

If you get a compilation error to the effect of `cannot find -lopenblas` then you either do not have openblas installed on your system, or else it is not installed in a place where `openblas-src` and `blas-src` can find it. Either check to make sure openblas is installed correctly, or else edit [`Cargo.toml`](./Cargo.toml), replacing `"openblas-system"` and `"system"` with `"openblas-static"` and `"static"`. This will compile a bundled version of the openblas and lapack source and statically link to it. Note that this workaround will dramatically increase compilation size, increase the size of the binary, and potentialy build a less-highly-optimized version of openblas than the one bundled with your system.

## Mathematical Background

Consider the linear regression problem $Y = X\beta + \epsilon$ where $Y$ is a column vector of observations, $X$ is a observations $\times$ predictor design matrix, and $\beta$ is a column vector of regression coefficients for each predictor.

$$
\underset{obs\times 1}{Y} = \left[\begin{matrix}
y_1 \\
y_2 \\
\vdots \\
y_o
\end{matrix}\right],\ \underset{obs\times pred}{X} = \left[\begin{matrix}
x_{11} & x_{22} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{o1} & x_{o2} & \cdots & x_{op}
\end{matrix}\right],\ \underset{pred\times 1}{\beta} = \left[\begin{matrix}
\beta_1 \\
\beta_2 \\
\vdots \\
\beta_p
\end{matrix}\right]
$$

### Ordinary Least Squares

Ordinary least squares (OLS) regression assumes identically and independently distributed errors $\epsilon\overset{iid}{\sim}\mathcal{N}(0,\sigma^2)$. The ordinary least squares estimator for $\beta$ is:

$$
\hat{\beta} = \left(X^\intercal X\right)^{-1}X^\intercal = X^+Y
$$

And the OLS estimate of the covariance of $\hat{\beta}$, which we will call $\Sigma$, is:

$$
\Sigma = \sigma^2\left(X^\intercal X\right)^{-1} = \sigma^2X^+\left(X^+\right)^\intercal
$$

The quadratic form using $\left(X^\intercal X\right)^{-1}$ is common in mathematical texts, but taking the inverse is numerically unstable when the matrix $X^\intercal X$ is ill-conditioned. The second solution expressed using $X^+$, the Moore-Penrose pseudoinverse of $X$, is more numerically stable.

### Random & Mixed Effects

The OLS assumption of _iid_ errors does not hold in many scientific problems. In particular:

- Repeated measures of the same subgroup may have correlated errors
- Measures from different subgroups may have heteroskedastic errors

In conventional linear regression, these "groups" or "clusters" of observations with non-_iid_ errors are modeled as random effects in a mixed effects model. However, solving linear mixed effects models is computationally expensive, requiring an iterative solution.

### Marginal Models & Sandwich Estimators

Here, we deal with a computational shortcut involving the _marginal model_. If we make the simplifying assumption that the random effects are not correlated with any of the predictors in $X$, then the marginal estimate of $\beta$ is the same as the OLS estimate above. However, the OLS estimate of $\Sigma$ will be too small; see [this comic example](https://xkcd.com/2533/).

> Note: If the random effect is correlated with a predictor in $X$ then $\hat{\beta}_{OLS}$ will be biased. For example, in a study with multiple sites, if participants at one site have greater levels of poverty, then we cannot control for poverty in the design matrix $X$.

We can adjust the calculation of $\Sigma$ using the so-called cluster-robust Huber-White sandwich estimator. The SwE is _asymptotically_ correct, that is, it converges on $\Sigma$ when the sample size is large. Its name comes from the structure of "meat" between two slices of "bread."

$$
\hat{\Sigma} = \overbrace{\left(X^\intercal X\right)^{-1}}^{\textrm{bread}} \underbrace{\left(X^\intercal VX\right)}_{\textrm{meat}} \overbrace{\left(X^\intercal X\right)^{-1}}^{\textrm{bread}} = X^+V\left(X^+\right)^\intercal
$$

The matrix $V$ is an $obs\times obs$ empirical covariance matrix obtained from the OLS residuals:

$$
\hat{\epsilon} = Y-X\hat{\beta}_{OLS}
$$

Typically $V$ is given a block-diagonal structure where each block corresponds to a "cluster" or "group" of correlated observations. Here, $\hat{\epsilon}_b$ is the rows of $\epsilon$ corresponsing to the observations in block $b$. The blocks do not have to be uniform in size; each cluster may have a different number of observations. There must be at least two blocks. Using $V=\epsilon\epsilon^\intercal$ leads to a degenerate solution.

$$
V = \left[\begin{matrix}
\boxed{\hat{\epsilon}_1\hat{\epsilon}_1^\intercal} & \mathbf{0} & \cdots & \mathbf{0} \\
\mathbf{0} & \boxed{\hat{\epsilon}_2\hat{\epsilon}_2^\intercal} & \cdots & \mathbf{0} \\
\vdots & \mathbf{0} & \ddots & \vdots \\
\mathbf{0} & \mathbf{0} & \cdots & \boxed{\hat{\epsilon}_b\hat{\epsilon}_b^\intercal}
\end{matrix}\right]
$$

> Note: For intuition, it can be shown that when $V=\sigma^2 I$ the SwE $\hat{\Sigma}$ is equal to the OLS $\hat{\Sigma}$.

### Computing SwE

Algorithmically, $V$ is a large $obs\times obs$ matrix whereas $\hat{\Sigma}$ is a much smaller $pred\times pred$ matrix. We can compute $\hat{\Sigma}$ without instantiating the entirety of $V$ in memory by summing each block's contribution. Let $X^+_b$ be the columns of $X^+$ corresponding to the observations in block $b$. It can be shown that:

$$
\underset{pred\times pred}{\hat{\Sigma}} = \sum_{i=1}^b X^+_i\epsilon_i\epsilon_i^\intercal\left(X^+_i\right)^\intercal
$$

Let $\mathcal{H}$ be the $pred\times obs$ "half sandwhich." Then we can express the SwE as the product of a matrix and its transpose, proving that it is positive semi-definite:

$$
\mathcal{H}_b=X^+_b\epsilon_b
$$

$$
\hat{\Sigma} = \sum_{i=1}^b \mathcal{H}_i \mathcal{H}_i^\intercal
$$

The algorithm for obtaining $\hat{\Sigma}$ is therefore a loop iterating over the diagonal blocks in $V$. For each block (cluster of observations), we compute the half sandwich $\mathcal{H}_b$, obtain $\hat{\Sigma}_b$ for that block, and add it to the grand total for $\hat{\Sigma}$.

### Multiple Features

In neuroimaging applications, it is common to have multiple columns, or _features_, in $Y$ where we want to fit the same model specified by the predictors in $X$ independently for each column in $Y$. Then, rather than being a column vector, $\beta$ will be a matrix with one column for each feature. Note that $\epsilon$ is now a $obs\times feat$ matrix.

$$
\underset{obs\times feat}{Y} = \left[\begin{matrix}
y_{11} & y_{22} & \cdots & y_{1m} \\
y_{21} & y_{22} & \cdots & y_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
y_{n1} & y_{n2} & \cdots & y_{nm}
\end{matrix}\right]
,\
\underset{obs\times pred}{X} = \left[\begin{matrix}
x_{11} & x_{22} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{matrix}\right]
,\
\underset{pred\times feat}{\beta} = \left[\begin{matrix}
\beta_{11} & \beta_{12} & \cdots & \beta_{1m} \\
\beta_{21} & \beta_{22} & \cdots & \beta_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
\beta_{p1} & \beta_{p2} & \cdots & \beta_{pm}
\end{matrix}\right]
$$

> Note: This scenario is often called "massively univariate" regression because, although the presence of multiple columns in $Y$ gives the appearance of a multivariate model, each feature is fit to its own, independent, univariate model.

Fortunately, the OLS solution for $\hat{\beta}$ with multiple features is _exactly the same_ as the equation for the OLS solution with one feature given above. This allows for very fast, vectorized computation of $\hat{\beta}$ and $\hat{\epsilon}$.

Unfortunately, the computation of $\hat{\Sigma}$ is not so easily vectorized since each feature has its own $\hat{\Sigma}$ computed from the corresponding column in $\hat{\epsilon}$. It may be helpful to think of $\hat{\Sigma}$ as having three dimensions: $pred \times pred \times feat$. The algorithm now has two loops:

- Iterating over the features, computing a separate $pred\times pred$ $\hat{\Sigma}$ for each feature
- Iterating over blocks, summing the contribution of each block to $\hat{\Sigma}$

### Wild Bootstrap

Due to $\hat{\Sigma}$ being only asymptotically correct, and for neuroimaging applications in general, it is common to perform some kind of permutation test or bootstrap rather than assume the test statistic (e.g. a $t$-statistic) has a canonical distribution. Note that many such randomization methods assume exchangeability between observations, an assumption that is violated by clustered observations. The wild bootstrap is a popular randomization method for clustered data that can be used with the SwE.

The introduction of a bootstrapping step now adds a third loop:

- Features
- Blocks
- Wild bootstrap resamples of $Y$

> Note: Typically we will not need to retain an entire $\hat{\Sigma}$ for each bootstrap. Instead, we would calculate a test statistic, such as a $t$-statistic, for each bootstrap, and just keep track of that.

### Notes on Inference

Given $\hat{\beta}$ and $\hat{\Sigma}$, the Student's $t$-statistic is given as follows.  It canonically has $obs - pred - 1$ degrees of freedom.

$$
t = \frac{\hat{\beta}}{\sqrt{\textrm{diag}\left(\hat{\Sigma}\right)}}
$$

And the Wald statistic is obtained as follows. Suppose that, under the null hypothesis, there are $q$ linear constraints on $\beta$.

$$
H_0:\ \underset{q\times pred}{R}\ \underset{pred\times 1}{\beta} = \underset{q\times 1}{r}
$$

For intuition, we can imagine:
$$
R = \left[\begin{matrix}1 & 0 & \cdots & 0\end{matrix}\right]
$$

$$
r = \left[\begin{matrix}0\end{matrix}\right]
$$

In which case we are performing a t-test on the first row of $\beta$.

In the massively univariate case the vector $r$ may be a matrix with $feat$ columns, although very often we want to apply the same constraint to all the columns of $\beta$, in which case we can leave $r$ as a $q\times 1$ vector and broadcast element-wise operations where appropriate.

$$
H_0:\ \underset{q\times pred}{R}\ \underset{pred\times feat}{\beta} = \underset{q\times feat}{r}
$$

The Wald statistic $W$ can be computed as follows (note the more numerically stable version using the pseudoinverse). When $\beta$ has $feat >1$ columns then only the diagonal of $W$, a vector of length $feat$, need be computed.

$$
W = \left(R\beta-r\right)^\intercal\left(R\Sigma_\hat{\beta}R^\intercal\right)^{-1}\left(R\beta-r\right)\\
W = \left[\left(RX^+\right)^+\left(R\beta-r\right)\right]^\intercal\left[\left(RX^+\right)^+\left(R\beta-r\right)\right]\\
$$

In the homoskedastic case, if the population variance is known (rarely the case) or $obs\rightarrow\infty$ then, asymptotically, $W$ has a $\chi^2$ distribution with $q$ degrees of freedom.  Otherwise $W$ has an $F$ distribution with degrees of freedom $d1=q,\ d2=obs-feat$.


## See Also

See my old example repository for a Rosetta Stone of sorts for Matlab to Rust: https://github.com/benkay86/matlab-ndarray-tutorial