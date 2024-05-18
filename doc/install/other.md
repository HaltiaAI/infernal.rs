---
sidebar_label: Others
sidebar_position: 3
---

# Installation from Source Code

## Prerequisites

- [Git]
- [GNU Make] 3.81+
- [Protobuf]
- [Rust] 1.70+

Firstly, make sure you have the full [Rust] toolchain installed.
This is best done via [Rustup]. We _do not_ recommend using Rust versions
already bundled with operating systems, as they are just likely to be too old.

## Cloning the Repository

```bash
git clone git@github.com:HaltiaAI/infernal.rs.git
```

## Building the Project

```bash
cargo install
```

[Git]: https://git-scm.com
[GNU Make]: https://www.gnu.org/software/make/
[Protobuf]: https://protobuf.dev
[Rust]: https://rust-lang.org
[Rustup]: https://rustup.rs
