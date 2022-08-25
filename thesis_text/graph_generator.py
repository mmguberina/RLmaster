from tbparse import SummaryReader
from os import makedirs

games = [
"BreakoutNoFrameskip-v4",
"EnduroNoFrameskip-v4",
"MsPacmanNoFrameskip-v4",
"PongNoFrameskip-v4",
"QbertNoFrameskip-v4",
"SeaquestNoFrameskip-v4",
"SpaceInvadersNoFrameskip-v4"
]

log_names = [
"default=rl_with_unsupervised_on_256_bottleneck_01",
#"raibow_rae_correct_grad",
#"rainbow_compression_rec_loss_big_enc_enc_01",
"rainbow_only_small_enc_01",
#"rainbow_only_small_enc_02",
"rl_only_on_rl_pretrained_encoders",
#"rl_with_pretrained_unsupervised_on_256_bottleneck_01",
"rl_with_pretrained_unsupervised_on_and_with_256_bottleneck_01"
#"small_enc_compressor_rainbow_lr_0001_01"
]

log_names_legend = {
        "default=rl_with_unsupervised_on_256_bottleneck_01": "parallel-fs-256",
        "raibow_rae_correct_grad": "123",
        "rainbow_compression_rec_loss_big_enc_enc_01": "parallel-fs-50",
        "rainbow_only_small_enc_01": "rl-only-small-net-1",
"rainbow_only_small_enc_02": "rl-only-small-net-2",
"rl_only_on_rl_pretrained_encoders": "rl-only-on-rl-pretrained",
"rl_with_pretrained_unsupervised_on_256_bottleneck_01":"rl-only-on-unsupervised-pretrained",
"rl_with_pretrained_unsupervised_on_and_with_256_bottleneck_01":"parallel-fs-50-pretrained",
"small_enc_compressor_rainbow_lr_0001_01":"parallel-small-net"
}

# TODO make a 1-1 correspondence between log_names
# and a meaningful name and use that to label lines
experiment_names = {}

hypothesis = "pretrained_vs_rl"
makedirs("/home/gospodar/chalmers/MASTER/RLmaster/thesis_text/figure/" + hypothesis, exist_ok=True)
dest_dir = "/home/gospodar/chalmers/MASTER/RLmaster/thesis_text/figure/" + hypothesis + "/"

logs = {}
for game in games:
    logs[game] = {}
    max_reward = 0
    min_reward = 10000
    for log_name in log_names:
        log_dir = "./" + game + "/" + log_name + "/"
        try:
            reader = SummaryReader(log_dir)
        except ValueError:
            print(game, log_dir, "is not a thing")
        df = reader.scalars
        try:
            steps = df[df.tag == "test/reward"]['step'].to_list()
            rewards = df[df.tag == "test/reward"]['value'].to_list()
        except AttributeError:
            print(log_name, "has some error (prolly empty)")

        logs[game][log_name] = {}
        logs[game][log_name]['steps'] = steps
        logs[game][log_name]['rewards'] = rewards

        max_r_this_log = max(rewards)
        min_r_this_log = min(rewards)
        if max_r_this_log > max_reward:
            max_reward = max_r_this_log
            #print(log_name)
        if min_r_this_log < min_reward:
            min_reward = min_r_this_log
    print(game, max_reward)
    print(game, min_reward)


top_tekst_breakout = \
"""
\\definecolor{blue}{RGB}{76,100,135}
\\definecolor{red}{RGB}{153,0,0}
\\definecolor{yellow}{RGB}{227,178,60}
\\definecolor{mycolor1}{rgb}{0.00000,0.44700,0.74100}%
\\definecolor{mycolor2}{rgb}{0.85000,0.32500,0.09800}%
\\definecolor{mycolor3}{rgb}{0.92900,0.69400,0.12500}%
%
\\begin{tikzpicture}

\\begin{axis}[%
title=Breakout,
width=10in,
height=5in,
%at={(2.596in,2.358in)},
% scale only axis,
xmin=0,
xmax=5000000,
xlabel style={font=\\color{white!15!black}},
xlabel={steps},
xlabel near ticks,
ymin=0,
ymax=400,
ylabel style={font=\\color{white!15!black}},
ylabel={Value},
ylabel near ticks,
ymajorgrids,
% scale=0.5,
scale=0.4,
axis background/.style={fill=white},
legend columns=2,
legend=south outside
]
"""

#legend style={legend cell align=left, align=left, draw=white!15!black}
top_tekst_enduro = \
"""
\\definecolor{blue}{RGB}{76,100,135}
\\definecolor{red}{RGB}{153,0,0}
\\definecolor{yellow}{RGB}{227,178,60}
\\definecolor{mycolor1}{rgb}{0.00000,0.44700,0.74100}%
\\definecolor{mycolor2}{rgb}{0.85000,0.32500,0.09800}%
\\definecolor{mycolor3}{rgb}{0.92900,0.69400,0.12500}%
%
\\begin{tikzpicture}

\\begin{axis}[%
title=Enduro,
% width=4.634in,
width=10in,
height=5in,
at={(2.596in,2.358in)},
% scale only axis,
xmin=0,
xmax=5000000,
xlabel style={font=\\color{white!15!black}},
xlabel={steps},
xlabel near ticks,
ymin=0,
ymax=2000,
ylabel style={font=\\color{white!15!black}},
ylabel={Value},
ylabel near ticks,
ymajorgrids,
% scale=0.5,
scale=0.4,
axis background/.style={fill=white},
legend style={legend cell align=left, align=left, draw=white!15!black}
]
"""

top_tekst_mspacman = \
"""
\\definecolor{blue}{RGB}{76,100,135}
\\definecolor{red}{RGB}{153,0,0}
\\definecolor{yellow}{RGB}{227,178,60}
\\definecolor{mycolor1}{rgb}{0.00000,0.44700,0.74100}%
\\definecolor{mycolor2}{rgb}{0.85000,0.32500,0.09800}%
\\definecolor{mycolor3}{rgb}{0.92900,0.69400,0.12500}%
%
\\begin{tikzpicture}

\\begin{axis}[%
title=MsPacman,
% width=4.634in,
width=10in,
height=5in,
at={(2.596in,2.358in)},
% scale only axis,
xmin=0,
xmax=5000000,
xlabel style={font=\\color{white!15!black}},
xlabel={steps},
xlabel near ticks,
ymin=-22,
ymax=3800,
ylabel style={font=\\color{white!15!black}},
ylabel={Value},
ylabel near ticks,
ymajorgrids,
% scale=0.5,
scale=0.4,
axis background/.style={fill=white},
legend style={legend cell align=left, align=left, draw=white!15!black}
]
"""

top_tekst_pong = \
"""
\\definecolor{blue}{RGB}{76,100,135}
\\definecolor{red}{RGB}{153,0,0}
\\definecolor{yellow}{RGB}{227,178,60}
\\definecolor{mycolor1}{rgb}{0.00000,0.44700,0.74100}%
\\definecolor{mycolor2}{rgb}{0.85000,0.32500,0.09800}%
\\definecolor{mycolor3}{rgb}{0.92900,0.69400,0.12500}%
%
\\begin{tikzpicture}

\\begin{axis}[%
title=Pong,
% width=4.634in,
width=10in,
height=5in,
at={(2.596in,2.358in)},
% scale only axis,
xmin=0,
xmax=5000000,
xlabel style={font=\\color{white!15!black}},
xlabel={steps},
xlabel near ticks,
ymin=-20,
ymax=20,
ylabel style={font=\\color{white!15!black}},
ylabel={Value},
ylabel near ticks,
ymajorgrids,
% scale=0.5,
scale=0.4,
axis background/.style={fill=white},
legend style={legend cell align=left, align=left, draw=white!15!black}
]
"""

top_tekst_qbert = \
"""
\\definecolor{blue}{RGB}{76,100,135}
\\definecolor{red}{RGB}{153,0,0}
\\definecolor{yellow}{RGB}{227,178,60}
\\definecolor{mycolor1}{rgb}{0.00000,0.44700,0.74100}%
\\definecolor{mycolor2}{rgb}{0.85000,0.32500,0.09800}%
\\definecolor{mycolor3}{rgb}{0.92900,0.69400,0.12500}%
%
\\begin{tikzpicture}

\\begin{axis}[%
title=Qbert,
% width=4.634in,
width=10in,
height=5in,
at={(2.596in,2.358in)},
% scale only axis,
xmin=0,
xmax=5000000,
xlabel style={font=\\color{white!15!black}},
xlabel={steps},
xlabel near ticks,
ymin=0,
ymax=16000,
ylabel style={font=\\color{white!15!black}},
ylabel={Value},
ylabel near ticks,
ymajorgrids,
% scale=0.5,
scale=0.4,
axis background/.style={fill=white},
legend style={legend cell align=left, align=left, draw=white!15!black}
]
"""

top_tekst_seaquest = \
"""
\\definecolor{blue}{RGB}{76,100,135}
\\definecolor{red}{RGB}{153,0,0}
\\definecolor{yellow}{RGB}{227,178,60}
\\definecolor{mycolor1}{rgb}{0.00000,0.44700,0.74100}%
\\definecolor{mycolor2}{rgb}{0.85000,0.32500,0.09800}%
\\definecolor{mycolor3}{rgb}{0.92900,0.69400,0.12500}%
%
\\begin{tikzpicture}

\\begin{axis}[%
title=Seaquest,
% width=4.634in,
width=10in,
height=5in,
at={(2.596in,2.358in)},
% scale only axis,
xmin=0,
xmax=5000000,
xlabel style={font=\\color{white!15!black}},
xlabel={steps},
xlabel near ticks,
ymin=0,
ymax=10500,
ylabel style={font=\\color{white!15!black}},
ylabel={Value},
ylabel near ticks,
ymajorgrids,
% scale=0.5,
scale=0.4,
axis background/.style={fill=white},
legend style={legend cell align=left, align=left, draw=white!15!black}
]
"""

top_tekst_spaceinvaders = \
"""
\\definecolor{blue}{RGB}{76,100,135}
\\definecolor{red}{RGB}{153,0,0}
\\definecolor{yellow}{RGB}{227,178,60}
\\definecolor{mycolor1}{rgb}{0.00000,0.44700,0.74100}%
\\definecolor{mycolor2}{rgb}{0.85000,0.32500,0.09800}%
\\definecolor{mycolor3}{rgb}{0.92900,0.69400,0.12500}%
%
\\begin{tikzpicture}

\\begin{axis}[%
title=SpaceInvaders,
width=10in,
height=5in,
at={(2.596in,2.358in)},
% scale only axis,
xmin=0,
xmax=5000000,
xlabel style={font=\\color{white!15!black}},
xlabel={steps},
xlabel near ticks,
ymin=0,
ymax=1500,
ylabel style={font=\\color{white!15!black}},
ylabel={Value},
ylabel near ticks,
ymajorgrids,
% scale=0.5,
scale=0.4,
axis background/.style={fill=white},
legend style={legend cell align=left, align=left, draw=white!15!black}
]
"""

#for game in logs:
#    print(game.keys())

logs['BreakoutNoFrameskip-v4']['top_text'] = top_tekst_breakout
logs['EnduroNoFrameskip-v4']['top_text'] = top_tekst_enduro
logs['MsPacmanNoFrameskip-v4']['top_text'] = top_tekst_mspacman
logs['PongNoFrameskip-v4']['top_text'] = top_tekst_pong
logs['QbertNoFrameskip-v4']['top_text'] = top_tekst_qbert
logs['SeaquestNoFrameskip-v4']['top_text'] = top_tekst_seaquest
logs['SpaceInvadersNoFrameskip-v4']['top_text'] = top_tekst_spaceinvaders

colors = ["blue", "red", "yellow", "mycolor1", "mycolor2", "mycolor3"]

for game in logs:
    fil = open(dest_dir + game[:-3] + ".tex", 'w')
    fil.write(logs[game]['top_text'])
    color_index = 0
    for log in logs[game]:
        if log == 'top_text':
            continue
        color_index += 1
#        fil.write(
#                """\\addplot [line width = 0.25mm]
#                  table[row sep=crcr]{
#                  """)
        fil.write(
                """\\addplot """) 
        fil.write(f"[color={colors[color_index]}, line width = 0.25mm]")
        fil.write("""
                table[row sep=crcr]{
                  """)
        for step, reward in zip(logs[game][log]['steps'], logs[game][log]['rewards']):
                fil.write(str(step) + " "  + str(reward) + "\\\\ \n")
        fil.write('};\n')
        fil.write('\\addlegendentry{' + log_names_legend[log] + '}\n')
    fil.write('\\end{axis}\n')
    fil.write('\\end{tikzpicture}')
    fil.close()

print(
"""
\\begin{figure}[!t]
  \\captionsetup[subfloat]{position=top,labelformat=empty}
  \\centering""")
for i, game in enumerate(logs):
    print("""
    \\subfloat[]{  \\resizebox{0.4\\textwidth}{!}{\\input{figure/""", end="")
    print(f"{hypothesis}/{game[:-3]}.tex}}}}}}",end="")
    if i % 2 == 1:
        print("""\\\\
  \\vspace{-1cm}""",end="")
print("""
  \\caption{caption text 23}
  \\label{fig:compare}
\\end{figure}
""")
