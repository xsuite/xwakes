import xwakes.wit as wit

re_wall_dipolar_x = wit.WakeComponentClassicThickWall(
    kind='dipolar_x',
    layer=wit.Layer(thickness=1e-7, dc_resistivity=1e6))

re_wall_dipolar_y = wit.WakeComponentClassicThickWall(
    kind='dipolar_y',
    layer=wit.Layer(thickness=1e-7, dc_resistivity=1e6))

re_wall_quadrupolar_x= wit.WakeComponentClassicThickWall(
    kind='quadrupolar_x',
    layer=wit.Layer(thickness=1e-7, dc_resistivity=1e6))

re_wall_quadrupolar_y = wit.WakeComponentClassicThickWall(
    kind='quadrupolar_y',
    layer=wit.Layer(thickness=1e-7, dc_resistivity=1e6))

re_wall = wit.Element(
    components=[re_wall_dipolar_x, re_wall_dipolar_y,
                re_wall_quadrupolar_x, re_wall_quadrupolar_y])


