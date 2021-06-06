#!/bin/sh -f

ninjatools/analyze_beam_pattern.py --yamlfile data/210511/param/20210512_HIMAC_BeamProfile_10spill_rot-m1.7deg_rect-1891pix_crop.yaml 

ninjatools/analyze_beam_pattern.py --yamlfile data/210511/param/20210512_HIMAC_BeamProfile_5spill_rot-0deg_rect-1891pix_crop.yaml 

ninjatools/analyze_beam_pattern.py --yamlfile data/210511/param/20210511_HIMAC_BeamProfile_10spill_rot-p1.1deg_rect-1891pix_crop.yaml

ninjatools/analyze_beam_pattern.py --yamlfile data/210511/param/20210511_HIMAC_BeamProfile_5spill_rot-m1.9deg_rect-1891pix_crop.yaml
