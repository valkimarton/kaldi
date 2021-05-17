#!/usr/bin/env bash
# Copyright      2017   David Snyder
#                2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#                2017   Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (mostly EERs) are inline in comments below.
#
# This example demonstrates a "bare bones" NIST SRE 2016 recipe using xvectors.
# It is closely based on "X-vectors: Robust DNN Embeddings for Speaker
# Recognition" by Snyder et al.  In the future, we will add score-normalization
# and a more effective form of PLDA domain adaptation.
#
# Pretrained models are available for this recipe.  See
# http://kaldi-asr.org/models.html and
# https://david-ryan-snyder.github.io/2017/10/04/model_sre16_v2.html
# for details.

# THESIS-DEBUG: For step-by-step ecexution (didn't work correctly)
# echo "Press CTRL+C to proceed."
# trap "pkill -f 'sleep 1h'" INT
# trap "set +x ; sleep 1h ; set -x" DEBUG 

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# SRE16 trials
# THESIS: own trials file
# sre16_trials=data/sre_combined/trials_NIST04_full
sre05_trials=data/sre2005_test_short/trials_nist05_short_common_unix
nnet_dir=exp/xvector_nnet_1a

stage=7
echo "Starting from stage $stage"

if [ $stage -le 0 ]; then
  # Path to some, but not all of the training corpora
  data_root=/home/ubuntu/combined_extracted_data
  nist04_root=$data_root/nist04
  nist05_train_root=$data_root/nist05_train_short
  nist05_test_root=$data_root/nist05_test_short

#  # Prepare telephone and microphone speech from Mixer6.
#  local/make_mx6.sh $data_root/LDC2013S03 data/

#  # Prepare SRE10 test and enroll. Includes microphone interview speech.
#  # NOTE: This corpus is now available through the LDC as LDC2017S06.
#  local/make_sre10.pl /export/corpora5/SRE/SRE2010/eval/ data/

#  # Prepare SRE08 test and enroll. Includes some microphone speech.
#  local/make_sre08.pl $data_root/LDC2011S08 $data_root/LDC2011S05 data/

#  # This prepares the older NIST SREs from 2004-2006.
#  local/make_sre.sh $data_root data/

  # THESIS: Prepare NIST04 dataset.
  # local/make_sre04_VRNT.sh $nist04_root data/

  # THESIS: Validate and Fix NIST05
  # utils/validate_data_dir.sh --no-text --no-feats data/sre2004
  # utils/fix_data_dir.sh data/sre2004

  # THESIS: Prepare NIST05 dataset.
  local/make_sre05_short_VRNT.sh $nist05_train_root $nist05_test_root data/
  
  # THESIS: Validate and Fix NIST05
  # utils/validate_data_dir.sh --no-text --no-feats data/sre2005_train_short
  # utils/fix_data_dir.sh data/sre2005_train_short
  utils/validate_data_dir.sh --no-text --no-feats data/sre2005_test_short
  utils/fix_data_dir.sh data/sre2005_test_short


#  # Combine all SREs prior to 2016 and Mixer6 into one dataset
#  utils/combine_data.sh data/sre \
#    data/sre2004 data/sre2005_train \
#    data/sre2005_test data/sre2006_train \
#    data/sre2006_test_1 data/sre2006_test_2 \
#    data/sre08 data/mx6 data/sre10
#  utils/validate_data_dir.sh --no-text --no-feats data/sre
#  utils/fix_data_dir.sh data/sre

#  # THESIS: Combine existing datasets from SREs prior to 2016 and Mixer6 into one dataset
 # utils/combine_data.sh data/sre \
 #   data/sre2004
 # utils/validate_data_dir.sh --no-text --no-feats data/sre
 # utils/fix_data_dir.sh data/sre

#  # Prepare SWBD corpora.
#  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
#    data/swbd_cellular1_train
#  local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
#    data/swbd_cellular2_train
#  local/make_swbd2_phase1.pl $data_root/LDC98S75 \
#    data/swbd2_phase1_train
#  local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
#    data/swbd2_phase2_train
#  local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
#    data/swbd2_phase3_train

#  # Combine all SWB corpora into one dataset.
#  utils/combine_data.sh data/swbd \
#    data/swbd_cellular1_train data/swbd_cellular2_train \
#    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

#  # Prepare NIST SRE 2016 evaluation data.
#  local/make_sre16_eval.pl /export/corpora5/SRE/R149_0_1 data

#  # Prepare unlabeled Cantonese and Tagalog development data. This dataset
#  # was distributed to SRE participants.
#  local/make_sre16_unlabeled.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data
fi

# THESIS: checkpoint
echo "Part 1: Data preparation COMPLETE"
# exit 1;

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl \
      /export/b{14,15,16,17}/$USER/kaldi-data/egs/sre16/v2/xvector-$(date +'%m_%d_%H_%M')/mfccs/storage $mfccdir/storage
  fi
  # THESIS: set used datasets. Oroginal was: (sre swbd sre16_eval_enroll sre16_eval_test sre16_major)
  # THESIS: set used datasets. My datasets: (sre sre2005_train sre2005_test sre2005_train_short sre2005_test_short)
  for name in sre sre2005_train sre2005_test_short; do
    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
  # THESIS - Here we should combine 'swbd' and 'sre' datasets to 'swbd_sre'. But right now we dont have swbd, so I only use sre
  # THESIS - Original was: utils/combine_data.sh --extra-files "utt2num_frames" data/swbd_sre data/swbd data/sre
  utils/combine_data.sh --extra-files "utt2num_frames" data/swbd_sre data/sre
  utils/fix_data_dir.sh data/swbd_sre
fi

# THESIS: checkpoint
# THESIS: --- OPTIONAL ---
echo "Part 2: Make MFCCs and compute the energy-based VAD for each dataset COMPLETE"
# exit 1;

# In this section, we augment the SWBD and SRE data with reverberation (visszhang),
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/swbd_sre/utt2num_frames > data/swbd_sre/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/swbd_sre data/swbd_sre_reverb
  cp data/swbd_sre/vad.scp data/swbd_sre_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/swbd_sre_reverb data/swbd_sre_reverb.new
  rm -rf data/swbd_sre_reverb
  mv data/swbd_sre_reverb.new data/swbd_sre_reverb

  # # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # # suitable for augmentation.
  # steps/data/make_musan.sh --sampling-rate 8000 /export/corpora/JHU/musan data

  # # Get the duration of the MUSAN recordings.  This will be used by the
  # # script augment_data_dir.py.
  # for name in speech noise music; do
  #   utils/data/get_utt2dur.sh data/musan_${name}
  #   mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  # done

  # # Augment with musan_noise
  # steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/swbd_sre data/swbd_sre_noise
  # # Augment with musan_music
  # steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/swbd_sre data/swbd_sre_music
  # # Augment with musan_speech
  # steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/swbd_sre data/swbd_sre_babble

  # Combine reverb, noise, music, and babble into one directory.
  # THESIS: Original was: utils/combine_data.sh data/swbd_sre_aug data/swbd_sre_reverb data/swbd_sre_noise data/swbd_sre_music data/swbd_sre_babble
  utils/combine_data.sh data/swbd_sre_aug data/swbd_sre_reverb # THESIS: modified line

  # # Take a random subset of the augmentations (128k is somewhat larger than twice
  # # the size of the SWBD+SRE list)
  # utils/subset_data_dir.sh data/swbd_sre_aug 128000 data/swbd_sre_aug_128k
  # utils/fix_data_dir.sh data/swbd_sre_aug_128k
  # THESIS: I have small amount of data --> Copy full data/swbd_sre_aug into data/swbd_sre_aug_128k
  cp -r data/swbd_sre_aug/* data/swbd_sre_aug_128k

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/swbd_sre_aug_128k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/swbd_sre_combined data/swbd_sre_aug_128k data/swbd_sre

  # Filter out the clean + augmented portion of the SRE list.  This will be used to
  # train the PLDA model later in the script.
  utils/copy_data_dir.sh data/swbd_sre_combined data/sre_combined
  utils/filter_scp.pl data/sre/spk2utt data/swbd_sre_combined/spk2utt | utils/spk2utt_to_utt2spk.pl > data/sre_combined/utt2spk
  utils/fix_data_dir.sh data/sre_combined

fi

# THESIS: checkpoint
echo "Part 3: Make MFCCs and compute the energy-based VAD for each dataset COMPLETE"
# exit 1;

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/swbd_sre_combined data/swbd_sre_combined_no_sil exp/swbd_sre_combined_no_sil
  utils/fix_data_dir.sh data/swbd_sre_combined_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/swbd_sre_combined_no_sil/utt2num_frames data/swbd_sre_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/swbd_sre_combined_no_sil/utt2num_frames.bak > data/swbd_sre_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/swbd_sre_combined_no_sil/utt2num_frames data/swbd_sre_combined_no_sil/utt2spk > data/swbd_sre_combined_no_sil/utt2spk.new
  mv data/swbd_sre_combined_no_sil/utt2spk.new data/swbd_sre_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/swbd_sre_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/swbd_sre_combined_no_sil/spk2num | utils/filter_scp.pl - data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2utt.new
  mv data/swbd_sre_combined_no_sil/spk2utt.new data/swbd_sre_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/utt2spk

  utils/filter_scp.pl data/swbd_sre_combined_no_sil/utt2spk data/swbd_sre_combined_no_sil/utt2num_frames > data/swbd_sre_combined_no_sil/utt2num_frames.new
  mv data/swbd_sre_combined_no_sil/utt2num_frames.new data/swbd_sre_combined_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/swbd_sre_combined_no_sil
fi

# THESIS: checkpoint
echo "Part 4: Preparing the features to generate examples for xvector training --> COMPLETE"
# exit 1;

# Trainign the DNN 
# if [ $stage -le 4 ]; then
#   local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage -1 \
#     --data data/swbd_sre_combined_no_sil --nnet-dir $nnet_dir \
#     --egs-dir $nnet_dir/egs
# fi

# THESIS: checkpoint
echo "Part 5: Training DNN --> SKIPPED"
# exit 1;


if [ $stage -le 7 ]; then
  # The SRE16 major is an unlabeled dataset consisting of Cantonese and
  # and Tagalog.  This is useful for things like centering, whitening and
  # score normalization.

  # THESIS: way of the original extraction
  # TODO: Remove
  # sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
  #   $nnet_dir data/sre_combined \
  #   exp/xvectors_sre16_major_original

  use_gpu=true

  # THESIS: changed data/sre16_major -> data/sre_combined
  # THESIS: NOT needed with pretrained models (DNN, PLDA, MEAN_VECTOR)
  # sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 16G" --nj 1 --use-gpu $use_gpu --cache-capacity 512 --chunk-size 500 \
  #   $nnet_dir data/sre_combined \
  #   exp/xvectors_sre16_major

  # Extract xvectors for SRE data (includes Mixer 6). We'll use this for
  # things like LDA or PLDA.
  # THESIS: NOT needed with pretrained models (DNN, PLDA, MEAN_VECTOR)
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 16G" --nj 1 --use-gpu $use_gpu --cache-capacity 512 --chunk-size 500 \
   $nnet_dir data/sre_combined \
   exp/xvectors_sre_combined

  # The SRE16 test data
  # THESIS: changed data/sre16_eval_test -> data/sre_combined
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 16G" --nj 1 --use-gpu $use_gpu --cache-capacity 512 --chunk-size 500 \
   $nnet_dir data/sre2005_test_short \
   exp/xvectors_sre2005_test_short

  # The SRE16 enroll data
  # THESIS: changed data/sre16_eval_enroll -> data/sre_combined
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 16G" --nj 1 --use-gpu $use_gpu --cache-capacity 512 --chunk-size 500 \
    $nnet_dir data/sre2005_train \
    exp/xvectors_sre2005_train
fi

# THESIS: checkpoint
echo "Part 6: x-vector extraction for different datasets --> COMPLETE"
# exit 1;

# THESIS: NOT needed with pretrained models (DNN, PLDA, MEAN_VECTOR)
if [ $stage -le 8 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd exp/xvectors_sre2005_train/log/compute_mean.log \
    ivector-mean scp:exp/xvectors_sre2005_train/xvector.scp \
    exp/xvectors_sre2005_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=150
  $train_cmd exp/xvectors_sre_combined/log/lda_own.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_sre_combined/xvector.scp ark:- |" \
    ark:data/sre_combined/utt2spk exp/xvectors_sre_combined/transform_own.mat || exit 1;

  # Train an out-of-domain PLDA model.
  $train_cmd exp/xvectors_sre_combined/log/plda_own.log \
    ivector-compute-plda ark:data/sre_combined/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_sre_combined/xvector.scp ark:- | transform-vec exp/xvectors_sre_combined/transform_own.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/xvectors_sre_combined/plda_own || exit 1;

  # Here we adapt the out-of-domain PLDA model to SRE05_train.
  $train_cmd exp/xvectors_sre2005_train/log/plda_own_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    exp/xvectors_sre_combined/plda_own \
    "ark:ivector-subtract-global-mean scp:exp/xvectors_sre2005_train/xvector.scp ark:- | transform-vec exp/xvectors_sre_combined/transform_own.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    exp/xvectors_sre2005_train/plda_own_adapt || exit 1;
fi

# THESIS: checkpoint
echo "Part 7: Centering, PLDA training --> COMPLETE"
echo "Part 7: Same dataset used for centering+normalization, PLDA training, enrolling, testing !!!"
# exit 1;

miss_stage_9=1
if [ $stage -le 9 ] && [ $miss_stage_9 -ne 1 ]; then
  # Get results using the out-of-domain PLDA model.
  # THESIS: slightly modofied
  # THESIS: removed Tagalog and Cantonese trials and scores, since it doesn't make sense in my dataset
  # THESIS: Uses Original DNN, PLDA and mean vector. (Mean vector is from xvector_sre_combined: Can be original or calculated from nist04!)
  $train_cmd exp/scores/log/sre2005_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/xvectors_sre2005_train/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/xvectors_sre_combined/plda - |" \
    "ark:ivector-mean ark:data/sre2005_train/spk2utt scp:exp/xvectors_sre2005_train/xvector.scp ark:- | ivector-subtract-global-mean exp/xvectors_sre_combined/mean.vec ark:- ark:- | transform-vec exp/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/xvectors_sre_combined/mean.vec scp:exp/xvectors_sre2005_test/xvector.scp ark:- | transform-vec exp/xvectors_sre_combined/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre05_trials' | cut -d\  --fields=1,2 |" exp/scores/sre05_eval_scores || exit 1;

  # utils/filter_scp.pl $sre16_trials_tgl exp/scores/sre16_eval_scores > exp/scores/sre16_eval_tgl_scores
  # utils/filter_scp.pl $sre16_trials_yue exp/scores/sre16_eval_scores > exp/scores/sre16_eval_yue_scores
  # pooled_eer=$(paste $sre16_trials exp/scores/sre16_eval_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  paste $sre05_trials exp/scores/sre05_eval_scores > exp/scores/sre05_eval_scores_for_eer_compute_full
  paste $sre05_trials exp/scores/sre05_eval_scores | awk '{print $6, $3}' > exp/scores/sre05_eval_scores_for_eer_compute
  pooled_eer=$(compute-eer exp/scores/sre05_eval_scores_for_eer_compute)
  compute-eer exp/scores/sre05_eval_scores_for_eer_compute > exp/scores/sre05_eval_eer_score
  # tgl_eer=$(paste $sre16_trials_tgl exp/scores/sre16_eval_tgl_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  # yue_eer=$(paste $sre16_trials_yue exp/scores/sre16_eval_yue_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Out-of-Domain PLDA, EER: Pooled ${pooled_eer}%"
  # EER: Pooled 11.73%, Tagalog 15.96%, Cantonese 7.52%
  # For reference, here's the ivector system from ../v1:
  # EER: Pooled 13.65%, Tagalog 17.73%, Cantonese 9.61%
fi

# THESIS: checkpoint
echo "Part 8: Getting result with out-of-domain PLDA --> COMPLETE"
echo "Part 8: Same dataset used for centering and enrolling!"
# exit 1;


if [ $stage -le 10 ]; then
  # Get results using the adapted PLDA model.
  # THESIS: removed Tagalog and Cantonese trials and scores, since it doesn't make sense in my dataset
  # THESIS: Uses Original DNN, adapted PLDA and enroll set mean vector.
  $train_cmd exp/scores/log/sre2005_eval_scoring_adapt_own.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:exp/xvectors_sre2005_train/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 exp/xvectors_sre2005_train/plda_own_adapt - |" \
    "ark:ivector-mean ark:data/sre2005_train/spk2utt scp:exp/xvectors_sre2005_train/xvector.scp ark:- | ivector-subtract-global-mean exp/xvectors_sre2005_train/mean.vec ark:- ark:- | transform-vec exp/xvectors_sre_combined/transform_own.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/xvectors_sre2005_train/mean.vec scp:exp/xvectors_sre2005_test_short/xvector.scp ark:- | transform-vec exp/xvectors_sre_combined/transform_own.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$sre05_trials' | cut -d\  --fields=1,2 |" exp/scores/sre05_eval_scores_adapt_own_short_layer7 || exit 1;

  # utils/filter_scp.pl $sre16_trials_tgl exp/scores/sre16_eval_scores_adapt > exp/scores/sre16_eval_tgl_scores_adapt
  # utils/filter_scp.pl $sre16_trials_yue exp/scores/sre16_eval_scores_adapt > exp/scores/sre16_eval_yue_scores_adapt
  paste $sre05_trials exp/scores/sre05_eval_scores_adapt_own_short_layer7 | awk '{print $6, $3}' > exp/scores/sre05_eval_scores_ADAPT_own_short_layer7_for_eer_compute
  pooled_eer=$(paste $sre05_trials exp/scores/sre05_eval_scores_adapt_own_short_layer7 | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  compute-eer exp/scores/sre05_eval_scores_ADAPT_own_short_layer7_for_eer_compute > exp/scores/sre05_eval_eer_score_ADAPTED_own_short_layer7
  # tgl_eer=$(paste $sre16_trials_tgl exp/scores/sre16_eval_tgl_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  # yue_eer=$(paste $sre16_trials_yue exp/scores/sre16_eval_yue_scores_adapt | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
  echo "Using Adapted PLDA, EER: Pooled ${pooled_eer}%"
  # EER: Pooled 8.57%, Tagalog 12.29%, Cantonese 4.89%
  # For reference, here's the ivector system from ../v1:
  # EER: Pooled 12.98%, Tagalog 17.8%, Cantonese 8.35%
  #
  # Using the official SRE16 scoring software, we obtain the following equalized results:
  #
  # -- Pooled --
  #  EER:          8.66
  #  min_Cprimary: 0.61
  #  act_Cprimary: 0.62
  #
  # -- Cantonese --
  # EER:           4.69
  # min_Cprimary:  0.42
  # act_Cprimary:  0.43
  #
  # -- Tagalog --
  # EER:          12.63
  # min_Cprimary:  0.76
  # act_Cprimary:  0.81
fi

# THESIS: checkpoint
echo "Part 9: Getting result with adapted PLDA --> COMPLETE"
echo "Part 9: Same dataset used for centering and enrolling!"
exit 1;

