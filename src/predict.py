import argparse
import os

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio

import config
import util
from mids_model import build_model


def get_wav_for_path_pipeline(path_names, sr):
    x = []
    signal_length = 0
    effects = [
                ["remix", "1"]
            ]
    if sr:
        effects.extend([
          #["bandpass", f"400",f"1000"],
          #["rate", f'{sr}'],
          ['gain', '-n'],
        ["highpass", f"200"]
        ])
    for path in path_names:
        signal, rate = librosa.load(path, sr=sr)
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(torch.tensor(signal).expand([2,-1]), sample_rate=rate, effects=effects)
        f = waveform[0]
        mu = torch.std_mean(f)[1]
        st = torch.std_mean(f)[0]
        # clip amplitudes
        signal = torch.clamp(f, min=mu-st*3, max=mu+st*3).unsqueeze(0)
        x.append(signal)
        signal_length += len(signal[0])/sr
    return x, signal_length


def get_output_file_name(
    root_out, filename, audio_format, step_size, threshold
):
    if filename.endswith(audio_format):
        output_filename = filename[
            :-4
        ]  # remove file extension for renaming to other formats.
    else:
        output_filename = filename  # no file extension present
    return (
        os.path.join(root_out, output_filename)
        + "_BNN_step_"
        + str(step_size)
        + "_"
        + f"{threshold:.1f}"
        + ".txt"
    )


# +
def write_output(
    data_path,
    predictions_path,
    audio_format,
    model_weights_path,
    det_threshold=np.arange(0.1, 1.1, 0.1),
    n_samples=10,
    feat_type="log-mel",
    n_feat=config.n_feat,
    win_size=config.win_size,
    step_size=config.win_size,
    n_hop=config.n_hop,
    sr=config.rate,
    norm_per_sample=config.norm_per_sample,
    batch_size=16,
    debug=False,
):
    model = build_model()
    model.load_weights(model_weights_path)
    print("Loaded model:", model_weights_path)

    mozz_audio_list = []

    print("Processing:", data_path, "for audio format:", audio_format)
    device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device)
    softmax = torch.nn.Softmax(dim=1)

    i_signal = 0
    with torch.no_grad():
        for root, dirs, files in os.walk(data_path):
            for filename in files:
                if audio_format not in filename:
                    continue
                print(root, filename)
                i_signal += 1
                #try:
                x, x_l = get_wav_for_path_pipeline(
                    [os.path.join(root, filename)], sr=sr
                )
                if debug:
                    print(filename + " signal length", x_l, (n_hop * win_size) / sr)
                if x_l < (n_hop * step_size) / sr:
                    print("Signal length too short, skipping:", x_l, filename)
                else:
                    #
                    frame_cnt = int(x[0].shape[1]//(step_size*n_hop))

                    pad_amt = int((win_size-step_size)*n_hop)
                    if pad_amt > 0:
                        pad_l = torch.zeros(1,pad_amt) + (0.1**0.5)*torch.randn(1, pad_amt)
                        pad_r = torch.zeros(1,pad_amt) + (0.1**0.5)*torch.randn(1, pad_amt)
                        X = torch.cat([pad_l,x[0],pad_r],dim=1).unfold(1,int(win_size*n_hop),int(step_size*n_hop)).transpose(0,1).to(device) # b, 1, s
                    else:
                        X = x[0].unfold(1,int(win_size*n_hop),int(step_size*n_hop)).transpose(0,1).to(device) # b, 1, s

                    out = []
                    X_CNN = []

                    preds_batch = []
                    spec_batch = []
#                     if filename == '206725.wav' or filename == '208065.wav':
#                         print('len')
#                         print(len(X.shape),X.shape)
#                         print(batch_size)

                    for X_batch in torch.split(X,batch_size,0):
                        preds = model(X_batch)
                        preds_prod = softmax(preds['prediction']).cpu().detach()
                        preds_batch.append(preds_prod)
                        spec_batch.append(preds['spectrogram'].cpu().detach().numpy())

                    out = torch.cat(preds_batch)
                    X_CNN.append(np.concatenate(spec_batch))

#                     print(f":{frame_cnt}")

                    p = torch.cat([out[i:frame_cnt+i,1:2] for i in range(win_size//step_size)],dim=-1).mean(dim=1).numpy()

                    b_out = np.array([torch.cat([out[i:frame_cnt+i,0:1],out[i:frame_cnt+i,1:2]],dim=1).numpy() for i in range(win_size//step_size)])

                    G_X, U_X, _ = util.active_BALD(np.log(b_out), frame_cnt, 2)



    #                 y_to_timestamp = np.repeat(np.mean(out, axis=0), step_size, axis=0)
    #                 G_X_to_timestamp = np.repeat(G_X, step_size, axis=0)
    #                 U_X_to_timestamp = np.repeat(U_X, step_size, axis=0)

                    root_out = root.replace(data_path, predictions_path)
                    print("dir_out", root_out, "filename", filename)

                    if not os.path.exists(root_out):
                        os.makedirs(root_out)

                    # Iterate over threshold for threshold-independent metrics
                    for th in det_threshold:
                        #####
                        true_indexes = np.where(p>th)[0]
                        # group by consecutive indexes
                        true_group_indexes = np.split(true_indexes, np.where(np.diff(true_indexes) != 1)[0]+1)

                        true_hop_indexes = np.where(p>th)[0]
                        # group by consecutive indexes
                        true_hop_group_indexes = np.split(true_hop_indexes, np.where(np.diff(true_hop_indexes) != 1)[0]+1)

                        preds_list = []
                        for hop_group in true_hop_group_indexes:
                            if len(hop_group) > 0:
                                row = []
                                row.append(hop_group[0]*step_size*n_hop/sr)
                                row.append((hop_group[-1]+1)*step_size*n_hop/sr)
                                p_str = "{:.4f}".format(p[hop_group].mean()) +\
                                    " PE: " + "{:.4f}".format(np.mean(G_X[hop_group])) +\
                                    " MI: " + "{:.4f}".format(np.mean(U_X[hop_group]))
                                row.append(p_str)
                                preds_list.append(row)

                        #####

                        if debug:
                            print(preds_list)
                            for times in preds_list:
                                mozz_audio_list.append(
                                    librosa.load(
                                        os.path.join(root, filename),
                                        offset=float(times[0]),
                                        duration=float(times[1]) - float(times[0]),
                                        sr=sr,
                                    )[0]
                                )

                        text_output_filename = get_output_file_name(
                            root_out,
                            filename,
                            audio_format,
                            step_size,
                            th,
                        )

                        np.savetxt(
                            text_output_filename,
                            preds_list,
                            fmt="%s",
                            delimiter="\t",
                        )
                    print("Processed:", filename)
#                 except Exception as e:
#                     print(
#                         "[ERROR] Unable to load {}, {} ".format(
#                             os.path.join(root, filename), e
#                         )
#                     )

    print("Total files of " + str(audio_format) + " format processed:", i_signal)
# -

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""This function writes the predictions of the model."""
    )
    parser.add_argument("--extension", default=".wav", type=str)
    parser.add_argument(
        "--norm",
        default=True,
        help="Normalise feature windows with respect to themselves.",
    )

    project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_folder_path = os.path.join(project_path, "models")
    parser.add_argument(
        "--model_weights_path",
        default=os.path.join(
            models_folder_path,
            "model_e0_2022_06_05_18_21_52.pth",
        ),
        type=str,
        help="Path to model weights.",
    )
    parser.add_argument(
        "--win_size", default=config.win_size, type=int, help="Window size."
    )
    parser.add_argument(
        "--step_size", default=config.win_size, type=int, help="Step size."
    )
    parser.add_argument(
        "--custom_threshold",
        help="Specify threshold between 0.0 and 1.0 for mosquito classification. Overrides default of ten threshold outputs.",
    )
    parser.add_argument(
        "--BNN_samples", default=10, type=int, help="Number of MC dropout samples."
    )

    args = parser.parse_args()

    extension = args.extension
    win_size = args.win_size
    step_size = args.step_size
    n_samples = args.BNN_samples
    norm_per_sample = args.norm
    model_weights_path = args.model_weights_path
    if args.custom_threshold is None:
        th = np.arange(0.1, 1.1, 0.1)
        print("no custom th detected")
    else:
        th = [args.custom_threshold]
        print("custom th detected")

    data_path = os.path.join(project_path, "data/audio/test")
    predictions_path = os.path.join(project_path, "data/predictions/test")

    write_output(
        data_path,
        predictions_path,
        extension,
        model_weights_path=model_weights_path,
        norm_per_sample=norm_per_sample,
        win_size=win_size,
        step_size=step_size,
        n_samples=n_samples,
        det_threshold=th,
    )

    for i, th in enumerate(th):
        df_list = []
        for filename in os.listdir(predictions_path):
            if filename.endswith(f"{th:.1f}" + ".txt"):
                df_pred = pd.read_csv(
                    os.path.join(predictions_path, filename),
                    sep="\t",
                    names=["onset", "offset", "event_label"],
                )
                filename = filename.split("_BNN_")[0]
                df_pred["event_label"] = "mosquito"
                df_pred["filename"] = filename
                df_list.append(df_pred)

        if len(df_list) > 0:
            pd.concat(df_list).to_csv(
                predictions_path + "/baseline_" + f"{th:.1f}" + ".csv",
                sep="\t",
                index=False,
            )
