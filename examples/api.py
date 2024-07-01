
WakeResonator(kind=['dipolar_x', 'dipolar_y'], ...)
WakeResonator(kind={'dipolar_x': 1.0, 'dipolar_y': 2.0}, ...)
WakeResonator(kind=Yokoya('circular')


# Elements (what we expose)
WakeClassicThickWall # e.g. WakeClassicThickWall(kind=['dipolar_x', 'dipolar_y']...)
WakeClassicThickWallYokoya
WakeClassicThickWallCircular
WakeClassicThickWallFlatHorizontal
WakeClassicThickWallFlatVertical
WakeClassicThickWallElliptic
WakeClassicThickWallRectangular

WakeSingleLayerResistiveWall
WakeSingleLayerResistiveWallYokoya
WakeSingleLayerResistiveWallCircular
WakeSingleLayerResistiveWallFlatHorizontal
WakeSingleLayerResistiveWallFlatVertical
WakeSingleLayerResistiveWallElliptic
WakeSingleLayerResistiveWallRectangular

WakeTaperSingleLayerRestsistiveWall
WakeTaperSingleLayerRestsistiveWallYokoya
WakeTaperSingleLayerRestsistiveWallCircular
WakeTaperSingleLayerRestsistiveWallFlatHorizontal
WakeTaperSingleLayerRestsistiveWallFlatVertical
WakeTaperSingleLayerRestsistiveWallElliptic
WakeTaperSingleLayerRestsistiveWallRectangular

WakeResonator
WakeResonatorYokoya
WakeResonatorCircular
WakeResonatorFlatHorizontal
WakeResonatorFlatVertical
WakeResonatorElliptic
WakeResonatorRectangular

WakeTable

# =======================================
# - The resistive wall ones have a length
# - All of these have components inside
# - All of these can be summed if they are in the same place

# =======================================
# To combine them in an impedance model we can do:
wake_model = WakeModel(
    betx_ref=1.0,
    bety_ref=1.0,
    elements=[
       xt.ModelElement(re_wall_pipe1, betx=2.0, bety=2.0),
       xt.ModelElement(re_wall_pipe2, betx=2.0, bety=2.0),
       xt.ModelElement(re_wall_kicker, betx=2.0, bety=2.0),
    ],
)