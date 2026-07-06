# CARE-4DD

This repository provides the core implementation of **CARE-4DD: Context-Aware Retrieval-Enhanced Heterogeneous Graph Network for Disease Diagnosis**.

CARE-4DD is designed for disease diagnosis based on electronic health records. The model first constructs a heterogeneous EMR graph with Patient, Drug, and Procedure nodes, then retrieves patient-centered diagnostic evidence using personalized PageRank, and finally encodes the retrieved evidence patch for disease prediction.

## Overview

CARE-4DD consists of three main components:

1. **Patient-centered PPR Evidence Retrieval**  
   Personalized PageRank is used to retrieve Top-K drug and procedure evidence nodes for each patient.

2. **Evidence Patch Construction**  
   The patient token, drug evidence tokens, and procedure evidence tokens are organized into a patient-centered evidence patch.

3. **Evidence Patch Encoding**  
   Token mixing and channel mixing are used to model local interactions among patient, drug, and procedure evidence tokens.

## Repository Structure

```text
CARE-4DD/
│
├── care4dd_hgconv_model.py      # CARE-4DD model definition
├── medical_ppr.py               # PPR evidence retrieval script
├── train_hgconv_direct.py       # Training script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files ignored by Git
└── README.md                    # Usage instructions
