import numpy as np
from numpy import dot
from numpy.linalg import norm

vector_1 = [194755.937, 78295.812, 0.0, 2944.949, 0.0, 0.0, 328487.937, 206969.625, 745233.5, 1794.705, 215853.156, 142495.031, 31960.443, 0.0, 0.0, 344602.062, 164495.75, 19701.718, 401373.093, 1378.023, 126984.32, 0.0, 0.0, 0.0, 0.0, 0.0, 289.224, 0.0, 206657.859, 95962.843, 1149396.5, 675042.75, 78766.835, 15499.032, 604363.062, 207589.875, 8082623.0, 6277704.5, 0.0, 83200.578, 200210.796, 200210.796, 20882.037, 3338.295, 0.0, 0.0, 0.0, 0.0, 0.0, 26201.373, 0.0, 0.0, 86000.359, 34810.328, 271539.25, 224289.125, 58229.136, 41050.3, 0.0, 22662.023, 154037.906, 132207.843, 5791037.0, 1769561.75, 76279.625, 301504.906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3869.214, 5290.024, 19205.912, 0.0, 0.0, 47671.617, 36345.492, 900.597, 0.0, 268807.0, 243829.14, 81844.984, 72356.179, 0.0, 0.0, 340489.75, 131188.843, 0.0, 0.0, 0.0, 791239.75, 0.0, 0.0, 53744.402, 4969.974, 62621.628, 1763.335, 152387.312, 102311.812, 298641.812, 298641.812, 55871548.0, 17217786.0, 521170.875, 9600.387, 0.0, 0.0, 212847.437, 135709.062, 0.0, 201280.39, 36955.8, 4542.389, 58648.652, 58648.652, 163375.5, 76268.578, 0.0, 0.0, 45449.656, 13829.55, 0.0, 613637.75, 1028132.187, 681364.312, 2186269.5, 364032.968, 0.0, 0.0, 0.0, 0.0, 950.7, 4988.194, 400.982, 628.696, 0.0, 239391.156, 4423.653, 4423.653, 1125395.375, 196807.046, 3328.176, 3328.176, 0.0, 0.0, 0.0, 0.0, 9459.206, 9459.206, 0.0, 0.0, 0.0, 475637.062, 0.0, 0.0, 0.0, 0.0, 22011.062, 24135.55, 0.0, 0.0, 0.0, 0.0, 365634.218, 40947.035, 0.0, 0.0, 35729.738, 2675.488, 0.0, 0.0, 577847.187, 577847.187, 267181.312, 9090.977, 807541.062, 807541.062, 35684.582, 0.0, 2530.617, 2530.617, 20341.07, 14601.845, 0.0, 0.0, 29010.242, 10842.695, 0.0, 12519.862, 0.0, 0.0, 6212188.5, 1938451.25, 31355.16, 10482.308, 187202.39, 2313.611, 55932.609, 52056.003, 0.0, 62292.218, 247092.218, 172440.546, 711447.75, 270352.375, 10329.904, 10329.904, 27013.623, 11494.142, 4169.312, 0.0, 1534864.375, 174208.375, 278525.75, 13864.182, 251636.625, 4117.501, 3983780.5, 462871.937, 267149.25, 124825.57, 138088.39, 118783.921, 0.0, 362.78, 89636.25, 82147.0, 1346626.75, 74809.242, 0.0, 0.0, 0.0, 0.0, 53457.261, 53457.261, 515793.312, 515793.312, 0.0, 0.0, 0.0, 0.0, 399559.218, 213073.812, 11859.236, 0.0, 880526.75, 643365.812, 5459742.0, 1102440.75, 400249.5, 400249.5, 0.0, 0.0, 0.0, 1764.721, 217012.203, 0.0, 11143.622, 11143.622, 83028.25, 47333.687, 228.488, 3073.666, 0.0, 0.0, 1944062.5, 505928.468, 250125.265, 225353.625, 209148.14, 20617.986, 39936.343, 39936.343, 62636.242, 0.0, 32041.982, 32041.982, 294767.625, 104290.703, 13150.359, 13150.359, 2204163.25, 817834.625, 48063.382, 13072.911, 0.0, 0.0, 0.0, 70921.468, 0.0, 4678.406, 0.0, 0.0, 14244461.0, 13914107.0, 33859.582, 2940.679, 0.0, 0.0, 497142.062, 45795.589, 725.438, 0.0, 32349.742, 0.0, 1412807.5, 152448.109, 67474.789, 0.0, 0.0, 0.0, 0.0, 5459.395, 0.0, 20944.439, 0.0, 0.0, 238801.578, 238801.578, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 92801.007, 92801.007, 0.0, 0.0, 583272.0, 332249.031, 0.0, 3537.9, 14390.374, 0.0, 0.0, 21494.521, 81953.109, 20511.419, 0.0, 9906.819, 113841.992, 119592.171, 470459.156, 112195.71, 0.0, 44643.0, 0.0, 43519.917, 1144188.75, 96798.281, 8917.121, 2266.885, 0.0, 26895.953, 0.0, 0.0, 1051193.75, 804800.187, 391926.75, 164211.39, 2101592.5, 225105.968, 0.0, 0.0, 42304.718, 10370.936, 0.0, 909911.437, 0.0, 0.0, 0.0, 0.0, 462192.312, 462192.312, 58718840.0, 22252844.0, 720.179, 5599.706, 205567.015, 81174.632, 53671.937, 53671.937, 178496.421, 2866.045, 345805.156, 186102.468, 0.0, 0.0, 0.0, 0.0, 4761568.5, 103279.304, 0.0, 28669.839, 113088.906, 113088.906, 359042.812, 53149.8, 1566556.25, 225966.64, 2849735.5, 1069639.625, 243330.625, 55776.46, 753174.0, 54279.39, 128459.125, 128459.125, 0.0, 0.0, 0.0, 0.0, 128819.281, 15510.154, 17525.115, 5115.821, 680152.562, 96131.046, 305842.562, 16378.441, 433539.937, 5007.986, 413680.937, 69428.07, 0.0, 6638.378, 1445435.375, 675143.875, 0.0, 0.0, 751768.375, 179674.843, 1179399.375, 298908.218, 1731199.5, 335101.593, 0.0, 0.0, 15955.675, 15955.675, 0.0, 4418.167, 41547.234, 41547.234, 59937.078, 59937.078, 4195.205, 0.0, 57002.253, 571162.687, 0.0, 0.0, 4505.522, 0.0, 0.0, 252352.687, 64962.382, 0.0, 0.0, 0.0, 2591930.75, 388790.875, 37668.902, 37668.902, 4390353.0, 972135.125, 0.0, 0.0, 22629.681, 14100.341, 246829.875, 210956.078, 44604.417, 44604.417, 0.0, 4704.164, 852260.812, 852260.812, 0.0, 0.0, 1183.156, 1183.156, 87327.062, 15335.57, 121138.773, 76917.812, 0.0, 6443.016, 47009.816, 15494.687, 1556071.125, 38308.238, 2377512.0, 174804.0, 691324.0, 691324.0, 1327398.875, 425095.843, 0.0, 0.0, 548958.875, 16770.248, 312057.5, 153038.89, 21662.576, 8758.828, 0.0, 0.0, 467031.812, 134447.796, 0.0, 0.0, 0.0, 0.0, 13962.402, 5153.871, 22254.667, 10766.515, 5077350.0, 79032.835, 0.0, 0.0, 1224.118, 0.0, 0.0, 0.0, 20933.498, 9317.387, 0.0, 0.0, 0.0, 0.0, 3832501.75, 367343.5, 0.0, 73.604, 0.0, 0.0, 197.284, 197.284, 0.0, 5417.531, 0.0, 0.0, 0.0, 1903.453, 0.0, 0.0, 1026007.25, 20848.859, 115313.859, 57148.929, 0.0, 0.0, 907789.062, 324909.0, 18334.259, 0.0, 2521.191, 2521.191, 0.0, 0.0, 0.0, 638.526, 0.0, 0.0, 122472.281, 0.0, 0.0, 0.0, 95070.89, 6999.231, 13429.77, 640407.187, 20620.64, 20282.5, 5176211.5, 698803.312, 26342.843, 26342.843, 111306.351, 111306.351, 968743.75, 862704.375, 5781218.0, 64285.121, 323221.937, 323221.937, 12253121.0, 67320.351, 43809.472, 43809.472, 0.0, 0.0, 3700845.0, 1674027.5, 0.0, 0.0, 0.0, 0.0, 0.0, 17620.91, 75257.468, 49546.859, 0.0, 0.0, 0.0, 0.0, 187393.218, 83510.703, 20699228.0, 6009869.0, 983935.375, 339408.0, 100419.375, 4532.252, 0.0, 0.0, 152976.875, 65027.75, 79351.351, 57077.011, 225912.437, 225912.437, 197566.843, 197566.843, 113029.078, 69925.648, 32728.978, 32728.978, 0.0, 0.0, 0.0, 0.0, 9356.974, 12700.765, 2428867.0, 1097528.875, 0.0, 0.0, 843042.5, 124023.453, 204386.593, 16568.3, 0.0, 0.0, 635339.125, 316838.281, 0.0, 0.0, 0.0, 0.0, 778754.625, 513588.406, 5596.068, 34107.007, 0.0, 0.0, 1224168.75, 288746.218, 1368.529, 494.441, 15412.337, 13013.283, 1115066.0, 798305.875, 91639.726, 40674.214, 570.626, 0.0, 550973.062, 36426.035, 0.0, 61621.882, 198722.531, 198722.531, 468710.156, 468710.156, 0.0, 9795.743, 49783.171, 41565.289, 1301494.5, 398154.75, 0.0, 58287.933, 50828.96, 38569.308, 0.0, 0.0, 0.0, 0.0, 129688.101, 35402.929, 0.0, 0.0, 0.0, 8167.879, 377962.937, 42840.07, 336362.062, 27979.47, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 33395.042, 0.0, 8252.669, 6388.198, 0.0, 0.0, 0.0, 7335.495, 0.0, 0.0, 0.0, 0.0, 0.0, 11442.859, 519687.281, 519687.281, 24275960.0, 7357558.0, 0.0, 0.0, 0.0, 0.0, 81711.796, 59677.746, 0.0, 0.0, 605054.0, 605054.0, 0.0, 189240.437, 146950.437, 146950.437, 0.0, 71510.656, 339992.0, 39179.226, 0.0, 3355945.75, 837077.312, 125301.484, 0.0, 0.0, 0.0, 0.0, 30376.55, 15655.409, 570079.75, 570079.75, 160247.781, 129951.531, 0.0, 0.0, 0.0, 219839.906, 437714.5, 55495.464, 0.0, 0.0, 47463.718, 47463.718, 0.0, 0.0, 470959.937, 470959.937, 707573.0, 315613.812, 108308.0, 47489.996, 5917.165, 5917.165, 651237.437, 651237.437, 0.0, 0.0, 0.0, 0.0, 86366.976, 7024.251, 1960315.125, 924439.875, 0.0, 0.0, 0.0, 0.0, 0.0, 1852385.75, 658.508, 493.339, 1099599.5, 1068105.5, 0.0, 0.0, 0.0, 0.0, 483380.75, 402547.625, 0.0, 0.0, 0.0, 0.0, 31743.751, 0.0, 408838.562, 208977.0, 0.0, 0.0, 1365709.375, 1029688.875, 12325412.0, 3350696.75, 60950.953, 0.0, 225850.625, 133617.968, 77710.687, 2219.667, 694036.5, 435827.062, 28953.386, 28953.386, 0.0, 0.0, 24490.062, 16661.736, 0.0, 0.0, 76955.375, 76955.375, 46794.367, 0.0, 21653.222, 0.0, 0.0, 0.0, 0.0, 0.0, 280954.187, 188159.765, 0.0, 0.0, 69802.742, 61767.152, 27368.263, 18506.302, 0.0, 0.0, 1016.685, 0.0, 609181.562, 609181.562, 0.0, 0.0, 87941.515, 87941.515, 0.0, 0.0, 0.0, 0.0, 221035.39, 0.0, 0.0, 0.0, 0.0, 0.0, 1666312.5, 529461.25, 0.0, 0.0, 200910.0, 0.0, 430269.375, 351746.75, 819590.0, 3134752.0, 0.0, 0.0, 0.0, 109785.734, 0.0, 0.0, 377572.687, 107817.796, 538306.625, 348992.593, 294938.968, 1349.703, 0.0, 1719223.5, 204019.468, 77580.968, 414133.312, 414133.312, 549557.125, 258929.703, 0.0, 183.478, 0.0, 0.0, 687360.937, 200013.781, 0.0, 0.0, 1460.653, 3156.92, 72711.335, 33296.285, 359418.25, 53674.968, 349908.687, 349908.687, 0.0, 0.0, 778362.812, 47001.136, 0.0, 17333.835, 431343.687, 431343.687, 242811.062, 130909.609, 559807.312, 559807.312, 482582.062, 0.0, 273339.5, 273339.5, 1248932.75, 489155.343, 5511.678, 4709.798, 0.0, 0.0, 0.0, 0.0, 33878.687, 33878.687, 3012.808, 3496.192, 496911.5, 455227.625, 0.0, 0.0, 0.0, 0.0, 274317.5, 274317.5, 977419.75, 732506.0, 0.0, 0.0, 69162.406, 69162.406, 0.0, 0.0, 1618124.0, 1071446.875, 0.0, 55103.171, 0.0, 0.0, 47189.199, 16271.063, 0.0, 0.0, 315281.125, 315281.125, 82941.171, 82941.171, 928638.5, 728537.187, 0.0, 7801.353, 3011484.0, 677189.125, 1505.38, 81209.414, 862737.625, 672125.125, 87716.773, 8831.458, 0.0, 463781.875, 794589.0, 39428.101, 0.0, 78952.593, 0.0, 0.0, 0.0, 503294.437, 0.0, 20825.056, 4379.213, 13726.94, 6702821.0, 2974254.75, 0.0, 456137.625, 0.0, 244.796, 0.0, 1512.669, 0.0, 0.0, 0.0, 355787.031, 368.409, 0.0]
vector_2 = [194755.937, 78295.812, 0.0, 2944.949, 0.0, 0.0, 328487.937, 206969.625, 745233.5, 1794.705, 215853.156, 142495.031, 31960.443, 0.0, 0.0, 344602.062, 164495.75, 19701.718, 401373.093, 1378.023, 126984.32, 0.0, 0.0, 0.0, 0.0, 0.0, 289.224, 0.0, 206657.859, 95962.843, 1149396.5, 675042.75, 78766.835, 15499.032, 604363.062, 207589.875, 8082623.0, 6277704.5, 0.0, 83200.578, 200210.796, 200210.796, 20882.037, 3338.295, 0.0, 0.0, 0.0, 0.0, 0.0, 26201.373, 0.0, 0.0, 86000.359, 34810.328, 271539.25, 224289.125, 58229.136, 41050.3, 0.0, 22662.023, 154037.906, 132207.843, 5791037.0, 1769561.75, 76279.625, 301504.906, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3869.214, 5290.024, 19205.912, 0.0, 0.0, 47671.617, 36345.492, 900.597, 0.0, 268807.0, 243829.14, 81844.984, 72356.179, 0.0, 0.0, 340489.75, 131188.843, 0.0, 0.0, 0.0, 791239.75, 0.0, 0.0, 53744.402, 4969.974, 62621.628, 1763.335, 152387.312, 102311.812, 298641.812, 298641.812, 55871548.0, 17217786.0, 521170.875, 9600.387, 0.0, 0.0, 212847.437, 135709.062, 0.0, 201280.39, 36955.8, 4542.389, 58648.652, 58648.652, 163375.5, 76268.578, 0.0, 0.0, 45449.656, 13829.55, 0.0, 613637.75, 1028132.187, 681364.312, 2186269.5, 364032.968, 0.0, 0.0, 0.0, 0.0, 950.7, 4988.194, 400.982, 628.696, 0.0, 239391.156, 4423.653, 4423.653, 1125395.375, 196807.046, 3328.176, 3328.176, 0.0, 0.0, 0.0, 0.0, 9459.206, 9459.206, 0.0, 0.0, 0.0, 475637.062, 0.0, 0.0, 0.0, 0.0, 22011.062, 24135.55, 0.0, 0.0, 0.0, 0.0, 365634.218, 40947.035, 0.0, 0.0, 35729.738, 2675.488, 0.0, 0.0, 577847.187, 577847.187, 267181.312, 9090.977, 807541.062, 807541.062, 35684.582, 0.0, 2530.617, 2530.617, 20341.07, 14601.845, 0.0, 0.0, 29010.242, 10842.695, 0.0, 12519.862, 0.0, 0.0, 6212188.5, 1938451.25, 31355.16, 10482.308, 187202.39, 2313.611, 55932.609, 52056.003, 0.0, 62292.218, 247092.218, 172440.546, 711447.75, 270352.375, 10329.904, 10329.904, 27013.623, 11494.142, 4169.312, 0.0, 1534864.375, 174208.375, 278525.75, 13864.182, 251636.625, 4117.501, 3983780.5, 462871.937, 267149.25, 124825.57, 138088.39, 118783.921, 0.0, 362.78, 89636.25, 82147.0, 1346626.75, 74809.242, 0.0, 0.0, 0.0, 0.0, 53457.261, 53457.261, 515793.312, 515793.312, 0.0, 0.0, 0.0, 0.0, 399559.218, 213073.812, 11859.236, 0.0, 880526.75, 643365.812, 5459742.0, 1102440.75, 400249.5, 400249.5, 0.0, 0.0, 0.0, 1764.721, 217012.203, 0.0, 11143.622, 11143.622, 83028.25, 47333.687, 228.488, 3073.666, 0.0, 0.0, 1944062.5, 505928.468, 250125.265, 225353.625, 209148.14, 20617.986, 39936.343, 39936.343, 62636.242, 0.0, 32041.982, 32041.982, 294767.625, 104290.703, 13150.359, 13150.359, 2204163.25, 817834.625, 48063.382, 13072.911, 0.0, 0.0, 0.0, 70921.468, 0.0, 4678.406, 0.0, 0.0, 14244461.0, 13914107.0, 33859.582, 2940.679, 0.0, 0.0, 497142.062, 45795.589, 725.438, 0.0, 32349.742, 0.0, 1412807.5, 152448.109, 67474.789, 0.0, 0.0, 0.0, 0.0, 5459.395, 0.0, 20944.439, 0.0, 0.0, 238801.578, 238801.578, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 92801.007, 92801.007, 0.0, 0.0, 583272.0, 332249.031, 0.0, 3537.9, 14390.374, 0.0, 0.0, 21494.521, 81953.109, 20511.419, 0.0, 9906.819, 113841.992, 119592.171, 470459.156, 112195.71, 0.0, 44643.0, 0.0, 43519.917, 1144188.75, 96798.281, 8917.121, 2266.885, 0.0, 26895.953, 0.0, 0.0, 1051193.75, 804800.187, 391926.75, 164211.39, 2101592.5, 225105.968, 0.0, 0.0, 42304.718, 10370.936, 0.0, 909911.437, 0.0, 0.0, 0.0, 0.0, 462192.312, 462192.312, 58718840.0, 22252844.0, 720.179, 5599.706, 205567.015, 81174.632, 53671.937, 53671.937, 178496.421, 2866.045, 345805.156, 186102.468, 0.0, 0.0, 0.0, 0.0, 4761568.5, 103279.304, 0.0, 28669.839, 113088.906, 113088.906, 359042.812, 53149.8, 1566556.25, 225966.64, 2849735.5, 1069639.625, 243330.625, 55776.46, 753174.0, 54279.39, 128459.125, 128459.125, 0.0, 0.0, 0.0, 0.0, 128819.281, 15510.154, 17525.115, 5115.821, 680152.562, 96131.046, 305842.562, 16378.441, 433539.937, 5007.986, 413680.937, 69428.07, 0.0, 6638.378, 1445435.375, 675143.875, 0.0, 0.0, 751768.375, 179674.843, 1179399.375, 298908.218, 1731199.5, 335101.593, 0.0, 0.0, 15955.675, 15955.675, 0.0, 4418.167, 41547.234, 41547.234, 59937.078, 59937.078, 4195.205, 0.0, 57002.253, 571162.687, 0.0, 0.0, 4505.522, 0.0, 0.0, 252352.687, 64962.382, 0.0, 0.0, 0.0, 2591930.75, 388790.875, 37668.902, 37668.902, 4390353.0, 972135.125, 0.0, 0.0, 22629.681, 14100.341, 246829.875, 210956.078, 44604.417, 44604.417, 0.0, 4704.164, 852260.812, 852260.812, 0.0, 0.0, 1183.156, 1183.156, 87327.062, 15335.57, 121138.773, 76917.812, 0.0, 6443.016, 47009.816, 15494.687, 1556071.125, 38308.238, 2377512.0, 174804.0, 691324.0, 691324.0, 1327398.875, 425095.843, 0.0, 0.0, 548958.875, 16770.248, 312057.5, 153038.89, 21662.576, 8758.828, 0.0, 0.0, 467031.812, 134447.796, 0.0, 0.0, 0.0, 0.0, 13962.402, 5153.871, 22254.667, 10766.515, 5077350.0, 79032.835, 0.0, 0.0, 1224.118, 0.0, 0.0, 0.0, 20933.498, 9317.387, 0.0, 0.0, 0.0, 0.0, 3832501.75, 367343.5, 0.0, 73.604, 0.0, 0.0, 197.284, 197.284, 0.0, 5417.531, 0.0, 0.0, 0.0, 1903.453, 0.0, 0.0, 1026007.25, 20848.859, 115313.859, 57148.929, 0.0, 0.0, 907789.062, 324909.0, 18334.259, 0.0, 2521.191, 2521.191, 0.0, 0.0, 0.0, 638.526, 0.0, 0.0, 122472.281, 0.0, 0.0, 0.0, 95070.89, 6999.231, 13429.77, 640407.187, 20620.64, 20282.5, 5176211.5, 698803.312, 26342.843, 26342.843, 111306.351, 111306.351, 968743.75, 862704.375, 5781218.0, 64285.121, 323221.937, 323221.937, 12253121.0, 67320.351, 43809.472, 43809.472, 0.0, 0.0, 3700845.0, 1674027.5, 0.0, 0.0, 0.0, 0.0, 0.0, 17620.91, 75257.468, 49546.859, 0.0, 0.0, 0.0, 0.0, 187393.218, 83510.703, 20699228.0, 6009869.0, 983935.375, 339408.0, 100419.375, 4532.252, 0.0, 0.0, 152976.875, 65027.75, 79351.351, 57077.011, 225912.437, 225912.437, 197566.843, 197566.843, 113029.078, 69925.648, 32728.978, 32728.978, 0.0, 0.0, 0.0, 0.0, 9356.974, 12700.765, 2428867.0, 1097528.875, 0.0, 0.0, 843042.5, 124023.453, 204386.593, 16568.3, 0.0, 0.0, 635339.125, 316838.281, 0.0, 0.0, 0.0, 0.0, 778754.625, 513588.406, 5596.068, 34107.007, 0.0, 0.0, 1224168.75, 288746.218, 1368.529, 494.441, 15412.337, 13013.283, 1115066.0, 798305.875, 91639.726, 40674.214, 570.626, 0.0, 550973.062, 36426.035, 0.0, 61621.882, 198722.531, 198722.531, 468710.156, 468710.156, 0.0, 9795.743, 49783.171, 41565.289, 1301494.5, 398154.75, 0.0, 58287.933, 50828.96, 38569.308, 0.0, 0.0, 0.0, 0.0, 129688.101, 35402.929, 0.0, 0.0, 0.0, 8167.879, 377962.937, 42840.07, 336362.062, 27979.47, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 33395.042, 0.0, 8252.669, 6388.198, 0.0, 0.0, 0.0, 7335.495, 0.0, 0.0, 0.0, 0.0, 0.0, 11442.859, 519687.281, 519687.281, 24275960.0, 7357558.0, 0.0, 0.0, 0.0, 0.0, 81711.796, 59677.746, 0.0, 0.0, 605054.0, 605054.0, 0.0, 189240.437, 146950.437, 146950.437, 0.0, 71510.656, 339992.0, 39179.226, 0.0, 3355945.75, 837077.312, 125301.484, 0.0, 0.0, 0.0, 0.0, 30376.55, 15655.409, 570079.75, 570079.75, 160247.781, 129951.531, 0.0, 0.0, 0.0, 219839.906, 437714.5, 55495.464, 0.0, 0.0, 47463.718, 47463.718, 0.0, 0.0, 470959.937, 470959.937, 707573.0, 315613.812, 108308.0, 47489.996, 5917.165, 5917.165, 651237.437, 651237.437, 0.0, 0.0, 0.0, 0.0, 86366.976, 7024.251, 1960315.125, 924439.875, 0.0, 0.0, 0.0, 0.0, 0.0, 1852385.75, 658.508, 493.339, 1099599.5, 1068105.5, 0.0, 0.0, 0.0, 0.0, 483380.75, 402547.625, 0.0, 0.0, 0.0, 0.0, 31743.751, 0.0, 408838.562, 208977.0, 0.0, 0.0, 1365709.375, 1029688.875, 12325412.0, 3350696.75, 60950.953, 0.0, 225850.625, 133617.968, 77710.687, 2219.667, 694036.5, 435827.062, 28953.386, 28953.386, 0.0, 0.0, 24490.062, 16661.736, 0.0, 0.0, 76955.375, 76955.375, 46794.367, 0.0, 21653.222, 0.0, 0.0, 0.0, 0.0, 0.0, 280954.187, 188159.765, 0.0, 0.0, 69802.742, 61767.152, 27368.263, 18506.302, 0.0, 0.0, 1016.685, 0.0, 609181.562, 609181.562, 0.0, 0.0, 87941.515, 87941.515, 0.0, 0.0, 0.0, 0.0, 221035.39, 0.0, 0.0, 0.0, 0.0, 0.0, 1666312.5, 529461.25, 0.0, 0.0, 200910.0, 0.0, 430269.375, 351746.75, 819590.0, 3134752.0, 0.0, 0.0, 0.0, 109785.734, 0.0, 0.0, 377572.687, 107817.796, 538306.625, 348992.593, 294938.968, 1349.703, 0.0, 1719223.5, 204019.468, 77580.968, 414133.312, 414133.312, 549557.125, 258929.703, 0.0, 183.478, 0.0, 0.0, 687360.937, 200013.781, 0.0, 0.0, 1460.653, 3156.92, 72711.335, 33296.285, 359418.25, 53674.968, 349908.687, 349908.687, 0.0, 0.0, 778362.812, 47001.136, 0.0, 17333.835, 431343.687, 431343.687, 242811.062, 130909.609, 559807.312, 559807.312, 482582.062, 0.0, 273339.5, 273339.5, 1248932.75, 489155.343, 5511.678, 4709.798, 0.0, 0.0, 0.0, 0.0, 33878.687, 33878.687, 3012.808, 3496.192, 496911.5, 455227.625, 0.0, 0.0, 0.0, 0.0, 274317.5, 274317.5, 977419.75, 732506.0, 0.0, 0.0, 69162.406, 69162.406, 0.0, 0.0, 1618124.0, 1071446.875, 0.0, 55103.171, 0.0, 0.0, 47189.199, 16271.063, 0.0, 0.0, 315281.125, 315281.125, 82941.171, 82941.171, 928638.5, 728537.187, 0.0, 7801.353, 3011484.0, 677189.125, 1505.38, 81209.414, 862737.625, 672125.125, 87716.773, 8831.458, 0.0, 463781.875, 794589.0, 39428.101, 0.0, 78952.593, 0.0, 0.0, 0.0, 503294.437, 0.0, 20825.056, 4379.213, 13726.94, 6702821.0, 2974254.75, 0.0, 456137.625, 0.0, 244.796, 0.0, 1512.669, 0.0, 0.0, 0.0, 355787.031, 368.409, 0.0]


vector_1 = np.array(vector_1)
vector_2 = np.array(vector_2)

var_sim =  dot(vector_1, vector_2) / (norm(vector_1) * norm(vector_2))
print("sim: ", var_sim)