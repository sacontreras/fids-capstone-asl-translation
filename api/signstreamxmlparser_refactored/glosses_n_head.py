#!/usr/bin/python

# -*- coding: utf-8 -*-
# $Id$

# Simple example showing how to display the glosses track of a signstream database,
# and head shake movements that coincide with specific glosses
from __future__ import absolute_import

import sys

import analysis.signstream as ss

if len(sys.argv) != 2:
  sys.stderr.write("Usage: showglosses.py <XML file>\n")
  sys.exit(1)

def format_headshake(head_movements):
  temp = []
  for hm in head_movements:
    (hs, he) = hm.get_timecodes()
    hstext = hm.get_text()
    temp.append("%s (%d-%d)" % (hstext, hs, he))
  return "headshake: " + ", ".join(temp)

db = ss.SignStreamDatabase.read_xml(sys.argv[1])

for participant in db.get_participants():
  print(str(participant))
  for token in participant.get_tokens("main gloss"):
    (start, end) = token.get_timecodes()
    text = token.get_text()
    head_movements = token.get_coinciding_tokens("hm: shake")
    if len(head_movements) > 0:
      head_str = "; " + format_headshake(head_movements)
    else:
      head_str = ""
    print("%6d-%6d %s%s" % (start, end, text, head_str))
