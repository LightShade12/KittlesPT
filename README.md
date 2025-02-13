<h1><u>KittlesPT</u> - A CUDA PathTracing renderer library</h2>
<hr>
<p>
A casual hobby project towards creating an interactive path-tracing renderer running on GTX GPUs.
</p>
<img src="assets/cover (7).png">

<table style="">
<tr>
<td>
<img src="assets/cover (1).png">
</td>
<td>
<img src="assets/cover (3).png">
</td>
</tr>
<tr>
<td>
<img src="assets/cover (6).png">
</td>
<td>
<img src="assets/cover (4).png">
</td>
</tr>
</table>
<h3><u>Features</u></h3>
<ul>
<li>PBR texture inputs for materials (GLTF 2.0 specifications parity)</li>
<li>Multiple Importance Sampling</li>
<li>Nishita Atmosphere model (Single-Scattering)</li>
<li>Bloom / Veil (Kawase)</li>
<li>Two Level Acceleration Structure (BVH2)</li>
<li>Runtime mesh transformations</li>
<li>Unified BSDF(Conductor & Opaque/Transparent Dielectrics)</li>
<li>Temporal Accumulation & Motion Vectors Generation</li>
<li>Physical camera controls (ISO, aperture,shutter speed etc)</li>
<li>Auto-exposure via Scene Histogram</li>
<li>AgX tonemapping</li>
</ul>

<h3><u>External Dependencies</u></h3>
<ul>
<li>CUDA 12.4+ SDK</li>
</ul>

<h3><u>Libraries used</u></h3>
<ul>
<li>GLAD - OpenGL loader</li>
<li>GLM - 3D maths library</li>
<li>Optional <b>(SampleApp)</b>
<ul>
<li>GLFW - OpenGL windowing library</li>
<li>Dear ImGUI - GUI library</li>
<li>stb - Image loading library</li>
<li>TinyGLTF - GLTF format load/parse library</li>
</ul>
</li>
</ul>

<h3><u>Building</u></h3>
<p>
<b>Cmake will be added in future.</b> Currently the solution is Visual Studio only.
The solution consists of two projects, the core <u>KittlesPT</u> static library project, and a <u>SampleApp</u> executable project demonstrating the use of KittlesPT renderer API.
</p>

<hr>