# UAV
Problem Difficulty Classification
| Difficulty Mode | Training Set | Testing Set |
|-----------------|--------------|-------------|
| **easy** | Even IDs: 0, 2, 4, ..., 54 (28 problems) | Odd IDs: 1, 3, 5, ..., 55 (28 problems) |
| **difficult** | Odd IDs: 1, 3, 5, ..., 55 (28 problems) | Even IDs: 0, 2, 4, ..., 54 (28 problems) |

*Note: When `difficulty` is 'all', both training and testing sets contain all problems (0-55).*

---

UAV provides 56 terrain-based landscapes as realistic Unmanned Aerial Vehicle(UAV) path planning problems, each of which is 30D. The objective is to select given number of path nodes (x,y,z coordinates) from the 3D space, so the the UAV could fly as shortly as possible in a collision-free way.

- **Paper**："[Benchmarking global optimization techniques for unmanned aerial vehicle path planning.](https://arxiv.org/abs/2501.14503)" arXiv preprint arXiv:2501.14503 (2025).
- **Code Resource**： [UAV](https://zenodo.org/records/12793991)
- **Problem Suite Composition:**

  1. A total of **56 benchmark instances**.

  2. Selected from **5,000 automatically generated terrains**, with **56 diverse terrains** carefully chosen for their complexity and real-world resemblance.

  3. The terrains include a variety of features such as plains, hills, steep slopes, and deep valleys.

  4. **Composed of two obstacle configurations:**
     - **15 cylindrical threats** (sparse scenario)

     - **30 cylindrical threats** (dense scenario)
  5. Threats are modeled as cylinders with varying heights and radii, simulating realistic obstacles like radar stations or missile sites.
  6. The UAV must navigate from a fixed **start** to a **goal point**, positioned at opposite corners of the terrain.
  7. Start and goal points are located in safe regions, outside of the threat zones.
  
- **Instance Composition:**
  - **28 terrains × 2 threat settings = 56 total benchmark instances**
  

## Objective function 

$$
F(X_i)=\sum_{k=1}^5{b_k\cdot F_k(X_i)}
$$

$b_1=5;b_2=1;b_3=10;b_4=1;b_5=1$

## **1. Path Length Cost**

Let $X_i$ be the flight path $i$, where each point along the path is $P_{ij}=(x_{ij}, y_{ij}, z_{ij})$. Let $S$ and $G$ denote the start and goal points, respectively.

The path length cost $F_1(X_i)$ is defined as:

$$
F_1(X_i)=\|\overrightarrow{P_{iS}P_{i1}}\| + \sum_{j=1}^{n-1} \|\overrightarrow{P_{ij}P_{i,j+1}}\| + \|\overrightarrow{P_{in}P_{iG}}\|
$$

## **2. Obstacle Avoidance Cost**

Let $k$ be the index of the $k$-th threat, with center $C_k$ and radius $R_k$.  
Let $T_k$ be the penalty from threat $k$, and $d_k$ the distance from the path segment to the center of threat $k$.  
Let $J_{\text{pen}}=10^4$ represent a large penalty for entering the dangerous zone.  
Assume the drone has diameter $D$, and $S$ is the safety distance threshold.

The penalty function is defined as:

$$
T_k\left( \overrightarrow{P_{ij}P_{i,j+1}} \right) =
\begin{cases}
0, & \text{if } d_k > S + D + R_k \\
(S + D + R_k) - d_k, & \text{if } D + R_k < d_k \le S + D + R_k \\
J_{\text{pen}}, & \text{if } d_k \le D + R_k \\
\end{cases}
$$

The total obstacle avoidance cost $F_2(X_i)$ is:

$$
F_2(X_i) = \sum_{j=1}^{n-1} \sum_{k=1}^K T_k\left( \overrightarrow{P_{ij}P_{i,j+1}} \right)
$$

## **3. Altitude Cost**

Let $h_{\text{max}}$ and $h_{\text{min}}$ be the maximum and minimum allowable flight altitudes.  Let $h_{ij}$ be the height of point $P_{ij}$ above the terrain.

The altitude cost $H_{ij}$ is defined as:

$$
H_{ij} =
\begin{cases}
\left| h_{ij} - \frac{h_{\text{max}} + h_{\text{min}}}{2} \right|, & \text{if } h_{\text{min}} \le h_{ij} \le h_{\text{max}} \\
\infty, & \text{otherwise} \\
\end{cases}
$$

The total altitude cost $F_3(X_i)$ is:

$$
F_3(X_i) = \sum_{j=1}^{n} H_{ij}
$$

## **4. Smoothness Cost**

Let $\theta_{ij}$ be the deflection angle at point $j$ and $\phi_{ij}$ the climb angle.  Assume $P'_{ij}$ are the projected 2D points (ignoring altitude).

The deflection angle is computed as:

$$
\theta_{ij} = \cos^{-1} \left( 
\frac{
\overrightarrow{P'_{i,j-1} P'_{ij}} \cdot \overrightarrow{P'_{ij} P'_{i,j+1}}}
{
\left\| \overrightarrow{P'_{i,j-1} P'_{ij}} \right\| \cdot 
\left\| \overrightarrow{P'_{ij} P'_{i,j+1}} \right\|}
\right)
$$

The climb angle is computed as:

$$
\phi_{ij} = \tan^{-1} \left( 
\frac{z_{i,j+1} - z_{ij}}{\left\| \overrightarrow{P'_{ij} P'_{i,j+1}} \right\|}
\right)
$$

The total smoothness cost $F_4(X_i)$ is:

$$
F_4(X_i) = \beta_1 \sum_{j=2}^{n-1} \theta_{ij} + \beta_2 \sum_{j=2}^{n-1} |\phi_{ij} - \phi_{i,j-1}|
$$

Where $\beta_1$ and $\beta_2$ are penalty coefficients.


## **5. Terrain Cost**

Let $H$ be the terrain height matrix, representing the ground elevation over the entire area. For a given path $X_i$, we insert $n$ interpolated points between each pair of adjacent path points $P_{ij}$ and $P_{i,j+1}$ to evaluate terrain safety.

Each interpolated point along the path segment is defined as:

$$
P^{\text{interp}}_{ijl} = \left(x^{\text{interp}}_{ijl},\ y^{\text{interp}}_{ijl},\ z^{\text{interp}}_{ijl}\right), \quad l=1, 2, \dots, n
$$

Let $H(x, y)$ denote the interpolated terrain height at coordinate $(x, y)$. The cost at each interpolated point is defined as:

$$
C_{ijl} =
\begin{cases}
0, & \text{if } z^{\text{interp}}_{ijl} > H(x^{\text{interp}}_{ijl},\ y^{\text{interp}}_{ijl}) \\
\infty, & \text{otherwise}
\end{cases}
$$

Then, the total terrain violation cost for path $X_i$ is given by:

$$
F_5(X_i) = \sum_{j=1}^{n-1} \sum_{l=1}^{n} C_{ijl}
$$

This ensures that all interpolated segments of the path are strictly above the terrain surface.
