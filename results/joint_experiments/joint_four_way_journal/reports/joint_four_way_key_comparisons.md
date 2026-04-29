# Key Supplement Comparisons

| Comparison | Metric | Mean Delta | CI Low | CI High | paired t p | Wilcoxon p | Effect |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| upper: attention/heuristic vs uncoordinated/heuristic | reward | 214.091 | 193.971 | 234.212 | 1.76272e-09 | 0.00195312 | improves |
| upper: attention/heuristic vs uncoordinated/heuristic | latency | -198545 | -321198 | -75891.8 | 0.00522033 | 0.00976562 | improves |
| upper: attention/heuristic vs uncoordinated/heuristic | energy | -8.18617e+06 | -1.21359e+07 | -4.23648e+06 | 0.00113837 | 0.00195312 | improves |
| upper: attention/heuristic vs uncoordinated/heuristic | deadline_satisfaction_rate | 0.110901 | 0.0509317 | 0.17087 | 0.00236414 | 0.00585938 | improves |
| upper: attention/heuristic vs uncoordinated/heuristic | fairness | 0.184838 | 0.129475 | 0.240201 | 3.49517e-05 | 0.00195312 | improves |
| upper: attention/heuristic vs uncoordinated/heuristic | offloading_ratio_local | -0.0105073 | -0.0258923 | 0.00487766 | 0.156754 | 0.322266 | regresses |
| upper: attention/heuristic vs uncoordinated/heuristic | offloading_ratio_cooperative | -0.061722 | -0.0913676 | -0.0320764 | 0.00110483 | 0.00390625 | regresses |
| upper: attention/heuristic vs uncoordinated/heuristic | offloading_ratio_mbs | 0.071546 | 0.0371856 | 0.105906 | 0.00110402 | 0.00195312 | regresses |
| upper: attention/heuristic vs uncoordinated/heuristic | mbs_load_ratio | 0.0575174 | 0.0297579 | 0.0852768 | 0.0011406 | 0.00390625 | regresses |
| lower: uncoordinated/oracle-guided vs uncoordinated/heuristic | reward | 3.2709 | 2.89322 | 3.64858 | 1.08919e-08 | 0.00195312 | improves |
| lower: uncoordinated/oracle-guided vs uncoordinated/heuristic | latency | 621.773 | 466.11 | 777.435 | 8.2647e-06 | 0.00195312 | regresses |
| lower: uncoordinated/oracle-guided vs uncoordinated/heuristic | energy | -1.61861e+07 | -1.83918e+07 | -1.39805e+07 | 4.66227e-08 | 0.00195312 | improves |
| lower: uncoordinated/oracle-guided vs uncoordinated/heuristic | deadline_satisfaction_rate | -0.000802082 | -0.00100602 | -0.000598145 | 9.37973e-06 | 0.00195312 | regresses |
| lower: uncoordinated/oracle-guided vs uncoordinated/heuristic | fairness | 3.10704e-05 | -1.2211e-07 | 6.22629e-05 | 0.0507289 | 0.0839844 | improves |
| lower: uncoordinated/oracle-guided vs uncoordinated/heuristic | offloading_ratio_local | 0.229244 | 0.218714 | 0.239773 | 2.94191e-12 | 0.00195312 | improves |
| lower: uncoordinated/oracle-guided vs uncoordinated/heuristic | offloading_ratio_cooperative | -0.0970122 | -0.120558 | -0.0734667 | 6.40681e-06 | 0.00195312 | regresses |
| lower: uncoordinated/oracle-guided vs uncoordinated/heuristic | offloading_ratio_mbs | -0.132232 | -0.14785 | -0.116613 | 1.32971e-08 | 0.00195312 | improves |
| lower: uncoordinated/oracle-guided vs uncoordinated/heuristic | mbs_load_ratio | -0.0686105 | -0.0803865 | -0.0568345 | 3.45024e-07 | 0.00195312 | improves |
| full: attention/oracle-guided vs uncoordinated/heuristic | reward | 212.396 | 186.941 | 237.851 | 1.51189e-08 | 0.00195312 | improves |
| full: attention/oracle-guided vs uncoordinated/heuristic | latency | -199986 | -321858 | -78114.6 | 0.00482924 | 0.00585938 | improves |
| full: attention/oracle-guided vs uncoordinated/heuristic | energy | -2.2763e+07 | -2.83836e+07 | -1.71425e+07 | 7.37882e-06 | 0.00195312 | improves |
| full: attention/oracle-guided vs uncoordinated/heuristic | deadline_satisfaction_rate | 0.11021 | 0.0514334 | 0.168986 | 0.00216883 | 0.00390625 | improves |
| full: attention/oracle-guided vs uncoordinated/heuristic | fairness | 0.187599 | 0.128507 | 0.246691 | 5.18517e-05 | 0.00195312 | improves |
| full: attention/oracle-guided vs uncoordinated/heuristic | offloading_ratio_local | 0.208716 | 0.190067 | 0.227366 | 1.12569e-09 | 0.00195312 | improves |
| full: attention/oracle-guided vs uncoordinated/heuristic | offloading_ratio_cooperative | -0.103957 | -0.130926 | -0.0769873 | 1.10517e-05 | 0.00195312 | regresses |
| full: attention/oracle-guided vs uncoordinated/heuristic | offloading_ratio_mbs | -0.104243 | -0.130094 | -0.0783921 | 7.64575e-06 | 0.00195312 | improves |
| full: attention/oracle-guided vs uncoordinated/heuristic | mbs_load_ratio | -0.04707 | -0.0675156 | -0.0266245 | 0.000558083 | 0.00390625 | improves |
| full vs upper-only: attention/oracle-guided vs attention/heuristic | reward | -1.69554 | -17.7877 | 14.3966 | 0.816947 | 0.625 | regresses |
| full vs upper-only: attention/oracle-guided vs attention/heuristic | latency | -1441.26 | -40937.6 | 38055.1 | 0.936018 | 1 | improves |
| full vs upper-only: attention/oracle-guided vs attention/heuristic | energy | -1.45769e+07 | -1.72898e+07 | -1.18639e+07 | 6.90291e-07 | 0.00195312 | improves |
| full vs upper-only: attention/oracle-guided vs attention/heuristic | deadline_satisfaction_rate | -0.000691293 | -0.0196403 | 0.0182577 | 0.936034 | 1 | regresses |
| full vs upper-only: attention/oracle-guided vs attention/heuristic | fairness | 0.00276057 | -0.004721 | 0.0102421 | 0.425489 | 0.492188 | improves |
| full vs upper-only: attention/oracle-guided vs attention/heuristic | offloading_ratio_local | 0.219224 | 0.211627 | 0.22682 | 2.34301e-13 | 0.00195312 | improves |
| full vs upper-only: attention/oracle-guided vs attention/heuristic | offloading_ratio_cooperative | -0.0422348 | -0.058796 | -0.0256736 | 0.000269762 | 0.00390625 | regresses |
| full vs upper-only: attention/oracle-guided vs attention/heuristic | offloading_ratio_mbs | -0.175789 | -0.189998 | -0.16158 | 4.61264e-10 | 0.00195312 | improves |
| full vs upper-only: attention/oracle-guided vs attention/heuristic | mbs_load_ratio | -0.104587 | -0.115174 | -0.0940005 | 3.40457e-09 | 0.00195312 | improves |
