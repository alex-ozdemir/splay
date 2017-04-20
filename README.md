# Code for Alex Ozdemir's Undergraduate Thesis

## Generalized Splay Implementation

There is an implementation of both Top-Down and Bottom-Up generalized splay in
the `bench/` directory. These implementations reference a representation of the
ruleset which can be build from a string of the form:

```
<path> <result>
LL     OXOXOXXX
```

The binary in this directory runs benchmarks for a class of rule sets which for
some natural $k$ map paths of up to length $k$ into results that have the target
at their root and otherwise optimally balance the nodes on the path.

## Top Splay

The `top-splay/` directory contains an implementation of Top Splay and Bottom Up
splay which track the deepest item to enable worst-case testing. It also
includes data from running the algorithms on trees of different sizes.

## Plots

In each directory, run `make plot.png` to plot.
