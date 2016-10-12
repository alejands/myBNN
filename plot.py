#!/usr/bin/env python
# -----------------------------------------------------------------------------
#  File: plot.py
#  Description: Make plots
#  Created:     28-Apr-2009 Harrison B. Prosper
# -----------------------------------------------------------------------------
import os
import sys
from histutil import *
from ROOT import *
from time import sleep
# -----------------------------------------------------------------------------
XNBINS = 50
XMIN = -2.0
XMAX = 2.0

YNBINS = 50
YMIN = -2.0
YMAX = 2.0
# -----------------------------------------------------------------------------


def main():
    gROOT.ProcessLine(open("testnet.cpp").read())
    setStyle()
    net = testnet();
    net.useLast(150)

    c = TCanvas("fig_sinxcosy", "", 0, 0, 800, 400)
    c.Divide(2, 1)

    h1 = mkhist2("h1", "x", "y", XNBINS, XMIN, XMAX, YNBINS, YMIN, YMAX)
    h2 = mkhist2("h2", "x", "y", XNBINS, XMIN, XMAX, YNBINS, YMIN, YMAX)

    xstep = (XMAX - XMIN) / XNBINS
    ystep = (YMAX - YMIN) / YNBINS
    for i in xrange(XNBINS):
        x = XMIN + (i + 0.5) * xstep
        for j in xrange(YNBINS):
            y = YMIN + (j + 0.5) * ystep

            v = vector('double')()
            v.push_back(x)
            v.push_back(y)

            z1 = net(v)
            z2 = sin(2*x) * cos(2*y)
            h1.Fill(x, y, z1)
            h2.Fill(x, y, z2)
    c.cd(1)
    h1.Draw("SURF2")

    c.cd(2)
    h2.Draw("SURF2")

    c.Update()
    c.SaveAs(".png")
    gApplication.Run()
# -----------------------------------------------------------------------------
main()
