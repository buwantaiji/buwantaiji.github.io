---
layout: page
title: Hello World - Vno
date: 2016-02-16 15:32:24.000000000 +09:00
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

[toc]
## This is a 2nd title.
#### What's this

[Vno Jekyll](https://github.com/onevcat/vno-jekyll) is a theme for [Jekyll](http://jekyllrb.com). It is a port of my Ghost theme [vno](https://github.com/onevcat/vno), which is originally developed from [Dale Anthony's Uno](https://github.com/daleanthony/uno).

#### Usage

```bash
$ git clone https://github.com/onevcat/vno-jekyll.git your_site
$ cd your_site
$ bundler install
$ bundler exec jekyll serve
```

```c
#include <stdio.h>
int main(int argc, char* argv[]) {
    printf("hello, world!\n");
    return 0;
}
```

$$
    \sin(x + y) = \sin(x) \cos(y) + \cos(x) \sin(y)
$$

$$
\mathbf{V}_1 \times \mathbf{V}_2 =  \begin{vmatrix}
\mathbf{i} & \mathbf{j} & \mathbf{k} \\
\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\
\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0 \\
\end{vmatrix}
$$

Your site with `Vno Jekyll` enabled should be accessible in http://127.0.0.1:4000.

For more information about Jekyll, please visit [Jekyll's site](http://jekyllrb.com).

#### Configuration

All configuration could be done in `_config.yml`. Remember you need to restart to serve the page when after changing the config file. Everything in the config file should be self-explanatory.

#### Background image and avatar

You could replace the background and avatar image in `assets/images` folder to change them.

#### Sites using Vno

[My blog](http://onevcat.com) is using `Vno Jekyll` as well, you could see how it works in real. There are some other sites using the same theme. You can find them below:

| Site Name    | URL                                                |
| ------------ | ---------------------------------------------------|
| OneV's Den   | [http://onevcat.com](http://onevcat.com)           |
| July Tang    | [http://blog.julytang.xyz](http://onevcat.com)     |
| Harry Lee    | [http://qiuqi.li](http://qiuqi.li)                 |

> If you happen to be using this theme, welcome to [send me a pull request](https://github.com/onevcat/vno-jekyll/pulls) to add your site link here. :)

#### License

Great thanks to [Dale Anthony](https://github.com/daleanthony) and his [Uno](https://github.com/daleanthony/uno). Vno Jekyll is based on Uno, and contains a lot of modification on page layout, animation, font and some more things I can not remember. Vno Jekyll is followed with Uno and be licensed as [Creative Commons Attribution 4.0 International](http://creativecommons.org/licenses/by/4.0/). See the link for more information.
