<p align="center">
  <img src="imgs/architect_logo.png" height=250/>
</p>

## If you want to design and analyze complex autonomous systems, Architect is your new best friend.

Architect is a tool for **automated**, **robust**, **co-design**. Let's break that down:

- **Automated**: How big does this motor need to be? What control gains should I use? Designing cyberphysical systems requires solving countless mini-optimization problems, most of which are tangled together in terms of how they affect system performance. Architect helps you automatically find design parameters that maximize your system's performance. Architect is not stealing your job as the system designer: it is just making your job easier, allowing you to focus on creativity and solving engineering problems.
- **Robust**: Solid engineering design requires not only optimizing for the expected, but also planning for the unexpected. Architect helps make sure that your designs are robust to changes in the environment or operating domain --- Architect helps your designs fly even when the winds change.
- **Co-design**: Instead of thinking about the software and control system design separately from the hardware design, Architect helps you optimize both at the same time (co-designing both the software and hardware). More importantly, Architect is a tool for improving the design process: it's your new co-designer!

To fill this role, Architect provides the following core features:

- A unified framework for specifying design and verification problems for autonomous systems (all in a [JAX](https://jax.readthedocs.io/en/latest/)-compatible way)
- High-performance algorithms for both optimization and verification
- Common building blocks for defining autonomous systems simulators (Kalman filters, temporal logic specifications, etc.).

## Modern systems are neither purely hardware nor purely software.

These so-called *cyberphysical* systems require new ways of thinking about engineering design. Not only do cyberphysical intertwine hardware and software considerations (think about how closely the control software for a drone depends on the flight hardware), but they often involve massive scale (think about scaling from one drone to a network of hundreds carrying out deliveries).

A traditional approach to designing these systems might start by defining high-level requirements for the system at large, then proceeding first to design the hardware, then build the software, and iterate back and forth until the right level of performance is achieved. In addition to this back and forth between designing different parts of the system, this approach also involves a *design-analysis cycle* --- working to improve the design, then testing it in simulation to see how it might perform, then going back to the design.

This approach simplifies the design process to reduce the burden on the designer, but does it reliably find the best design?

## Try it for yourself

To install Architect, run the following commands:

```bash
git clone https://github.com/dawsonc/architect_private
cd architect_private
conda create -n architect_env python=3.9
conda activate architect_env
pip install -e .
pip install -r requirements.txt
```

To install the environment for use in Jupyter notebooks, run
```bash
python -m ipykernel install --user --name=architect_env
```

Hardware experiments currently require `rospy` (must be installed separately) and other dependencies specified in `requirements_hw.txt`

## Let's see an example

TODO: define a new example for V2.0

## Code organization

TODO

## Warning: Research code may contain sharp edges!

Architect is under active development as part of research in the [Reliable Autonomy Laboratory](realm.mit.edu) at MIT (REALM). We will aim to release a tagged version to coincide with each new publication and major advance in capabilities, but we currently cannot commit to a stable interface.

Architect was developed using Python on Ubuntu 20.04. Other operating systems are not officially supported.
