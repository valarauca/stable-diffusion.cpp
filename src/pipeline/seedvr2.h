#ifndef __SD_PIPELINE_SEEDVR2_H__
#define __SD_PIPELINE_SEEDVR2_H__

#include "stable-diffusion.h"

class StableDiffusionGGML;

namespace sd::pipeline {

    // Restores sd_vid_gen_params->ref_videos[0] with SeedVR2's one-step
    // restoration and returns the restored frames and the source audio.
    bool restore_seedvr2_video(StableDiffusionGGML* sd,
                               const sd_vid_gen_params_t* sd_vid_gen_params,
                               sd_image_t** frames_out,
                               int* num_frames_out,
                               sd_audio_t** audio_out,
                               int* fps_out);

}  // namespace sd::pipeline

#endif  // __SD_PIPELINE_SEEDVR2_H__
