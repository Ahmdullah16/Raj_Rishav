"""Raj Rishav Human Activity Recognition dashboard.

Deploy on Streamlit Community Cloud with entrypoint ``streamlit.py``.
Dataset metadata is public, but raw Kaggle videos remain on the owner's Google
Drive; inference requires an actual trained .keras checkpoint and classes.json.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pandas as pd
import streamlit as st

from har.app_support import (
    MODEL_VARIANTS, SPLIT_NAMES, dataframe_csv_bytes, display_label,
    get_dataset_summary, load_classes, load_split_table, matrix_from_csv,
    read_json_classes, read_optional_csv, read_optional_json, size_label,
)
from har.data import LABELS
from har.video import resolve_video_file, sample_video, validate_video

# Absolute repository-relative paths: Streamlit Community Cloud runs from repo root.
DATA_CSV = ROOT / 'data/train_val_test_splits_70_15_15.csv'
DATA_JSON = ROOT / 'data/dataset_summary_70_15_15.json'
MODEL_DIR = ROOT / 'artifacts'
RESULTS_DIR = ROOT / 'outputs'
GITHUB = 'https://github.com/Ahmdullah16/Raj_Rishav'
COLAB = 'https://colab.research.google.com/github/Ahmdullah16/Raj_Rishav/blob/main/notebooks/HAR_CNN_LSTM_Complete.ipynb'
DRIVE = 'https://drive.google.com/drive/folders/1EIboVuDhhsE_qL27u3u6QrzNyuAhWfbD'
SOURCE = 'https://www.kaggle.com/datasets/sharjeelmazhar/human-activity-recognition-video-dataset'

st.set_page_config(page_title='Human Activity Recognition | CNN + LSTM', page_icon='🎬', layout='wide', initial_sidebar_state='expanded')

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif;}
.stApp {background:radial-gradient(ellipse at top right,rgba(48,104,178,.10),transparent 44%),#0a1323;color:#eaf2fb}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.025em}
[data-testid="stSidebar"] {background:#101e34;border-right:1px solid #233955}
[data-testid="stSidebar"] * {color:#dceafb}
.block-container{padding-top:1.6rem;padding-bottom:4rem;max-width:1400px;}
.har-hero{background:linear-gradient(110deg,#122c50 0%,#173c62 58%,#134946 100%);border:1px solid #285078;border-radius:22px;padding:32px 35px;margin-bottom:22px;box-shadow:0 12px 40px rgba(0,0,0,.12)}
.har-hero .eyebrow{letter-spacing:.18em;color:#91e6da;font-size:.7rem;font-weight:700;text-transform:uppercase;margin-bottom:9px}
.har-hero h1{margin:0 0 10px 0;font-size:2.2rem;color:#fff}
.har-hero p{max-width:850px;color:#c6d9ed;margin:0;font-size:1.02rem;line-height:1.65}
.har-pill{display:inline-block;font-size:.75rem;margin-top:16px;color:#a1f3cf;background:#134842;border:1px solid #2f917c;border-radius:50px;padding:6px 12px}
.har-card{background:#14253c;border:1px solid #2b405b;border-radius:16px;padding:20px;margin:4px 0 16px;min-height:116px}
.har-card .title{font-size:.76rem;text-transform:uppercase;letter-spacing:.1em;color:#91a8c4;margin-bottom:8px}
.har-card .value{font-size:1.85rem;font-weight:700;color:#f7fbff;letter-spacing:-.025em}
.har-card .note{color:#a9bdd3;font-size:.81rem;margin-top:5px}
.har-step{background:#14253c;border:1px solid #2b405b;border-radius:12px;padding:17px;text-align:center;font-weight:600}
.har-arrow{text-align:center;color:#47ccbe;font-size:1.3rem;margin:0}
.har-caption{font-size:.86rem;color:#9eb5cf}
[data-testid="stMetric"]{background:#14253c;border:1px solid #2b405b;border-radius:14px;padding:15px}
[data-testid="stDataFrame"], [data-testid="stPlotlyChart"]{border-radius:12px}
.stButton>button[kind="primary"]{background:#16a698;border-color:#16a698;color:#071a27;font-weight:700}
a{color:#7adcd0!important}
hr{border-color:#2e4560}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def card(title: str, value: str, note: str = '') -> None:
    st.markdown(f'<div class="har-card"><div class="title">{title}</div>'
                f'<div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)


def hero(eyebrow: str, title: str, subtitle: str, pill: str = '') -> None:
    st.markdown('<div class="har-hero"><div class="eyebrow">' + eyebrow + '</div><h1>' + title +
                '</h1><p>' + subtitle + '</p>' +
                (f'<span class="har-pill">{pill}</span>' if pill else '') + '</div>', unsafe_allow_html=True)


def render_plot(fig) -> None:
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(14,27,44,.6)', font_color='#dceafb',
                      margin=dict(l=15,r=15,t=45,b=20), height=400)
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def get_split():
    rows = load_split_table(DATA_CSV)
    return rows, get_dataset_summary(rows)


def artifacts(selected_checkpoint=None):
    # Match visible training history/provenance to the checkpoint actually deployed.
    paths = [RESULTS_DIR, MODEL_DIR]
    def find(name):
        for base in paths:
            if (base / name).is_file():
                return base / name
        return paths[0] / name
    nine = bool(selected_checkpoint and selected_checkpoint.name == 'best_epoch9.keras')
    return {'history': find('history_epoch9.csv' if nine else 'history.csv'),
            'metrics': find('test_metrics.json'),
            'matrix': find('confusion_matrix.csv'),
            'meta': find('run_metadata_epoch9.json' if nine else 'run_metadata.json'),
            'comparison': find('model_comparison.csv')}


@st.cache_resource(show_spinner=False)
def resolve_checkpoint():
    from har.checkpoint import verify_published_checkpoint
    for path in (MODEL_DIR/'best_epoch9.keras', MODEL_DIR/'best.keras',
                 RESULTS_DIR/'best.keras', MODEL_DIR/'final.keras'):
        if path.is_file():
            # Public bundled versions are immutable and cryptographically pinned.
            if path.parent == MODEL_DIR and not verify_published_checkpoint(path):
                st.warning(f'Bundled {path.name} failed integrity verification; not loading.')
                continue
            for classes in (path.parent/'classes.json', MODEL_DIR/'classes.json', RESULTS_DIR/'classes.json'):
                if classes.is_file():
                    return path, classes
    # Recover only the verified checkpoint, not the original 1.86 GB dataset.
    from har.checkpoint import materialize_checkpoint
    try:
        return materialize_checkpoint(MODEL_DIR)
    except Exception as exc:
        st.session_state['checkpoint_download_error'] = str(exc)
        return None, None


@st.cache_resource(show_spinner=False)
def load_checkpoint(path: str, modified_ns: int):
    # TensorFlow is optional until a real checkpoint is available.
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False, safe_mode=True)


def probability_dataframe(values: np.ndarray, classes: list[str]) -> pd.DataFrame:
    result = pd.DataFrame({'Activity': [display_label(x) for x in classes],
                           'Probability': values.astype(float)})
    return result.sort_values('Probability', ascending=False).reset_index(drop=True)

try:
    rows, summary = get_split()
    frame = pd.DataFrame(rows)
    frame['bytes'] = pd.to_numeric(frame['bytes'], errors='coerce').fillna(0).astype('int64')
except Exception as error:
    st.error(f'Unable to read bundled dataset metadata: {error}')
    st.stop()

model_path, labels_path = resolve_checkpoint()
result_files = artifacts(model_path)
metrics = read_optional_json(result_files['metrics'])
metadata = read_optional_json(result_files['meta'])

with st.sidebar:
    st.markdown('## 🎬 HAR Lab')
    st.caption('IIT Guwahati • CNN + LSTM')
    st.divider()
    menu = ['Predict Activity', 'Overview', 'Dataset & Splits', 'Results',
            'Compare Models', 'Sensitivity & Diagnostics', 'Concepts & Mathematics', 'Downloads', 'About']
    page = st.radio('Explore', menu, label_visibility='collapsed', key='page')
    st.divider()
    if model_path:
        st.success('Verified CNN–LSTM checkpoint available')
        model_version = '9-epoch snapshot' if model_path.name == 'best_epoch9.keras' else 'original 3-epoch snapshot'
        st.caption(f'{model_version} • experimental recognition; held-out accuracy not measured.')
    else:
        st.info('Verified checkpoint could not be recovered automatically. Check artifact access or upload it on the Prediction page.')
        if st.session_state.get('checkpoint_download_error'):
            st.caption('Artifact recovery: ' + st.session_state['checkpoint_download_error'])
    st.caption(f'{summary["total_videos"]} videos • {summary["classes"]} classes')
    st.link_button('Open GitHub repository', GITHUB, use_container_width=True)

if page == 'Overview':
    hero('Video intelligence · Research project', 'Human Activity Recognition',
         'Recognize human activities from sequences of video frames by combining CNN spatial feature extraction with LSTM temporal learning.',
         'Seven real activities • Stratified 70 / 15 / 15')
    a,b,c,d = st.columns(4)
    with a: card('Dataset size', str(summary['total_videos']), 'Real videos from the Kaggle subset')
    with b: card('Activity classes', str(summary['classes']), 'One label per source-video folder')
    with c: card('Dataset storage', size_label(summary['total_bytes']), 'Video bytes in Google Drive')
    with d: card('Evaluation status', 'Available' if metrics else 'Awaiting run', 'No fabricated performance metrics')
    st.subheader('End-to-end workflow')
    for label in ['Video input · seven classes', '70 / 15 / 15 stratified splitting',
                  '20 uniformly sampled RGB frames (64 × 64)',
                  'TimeDistributed CNN · spatial embeddings', 'LSTM · temporal dependencies',
                  'Softmax probabilities · activity label']:
        st.markdown('<div class="har-step">' + label + '</div>', unsafe_allow_html=True)
        if 'Softmax' not in label:
            st.markdown('<div class="har-arrow">↓</div>', unsafe_allow_html=True)
    st.markdown('')
    st.info('The earlier project PDF reported results on **UCF101 (101 classes)**. Those results are not comparable to this seven-class Kaggle dataset and are not used as current results.')
    st.subheader('Project actions')
    c1,c2,c3=st.columns(3)
    with c1: st.link_button('Open Colab training notebook', COLAB, use_container_width=True)
    with c2: st.link_button('View dataset folder', DRIVE, use_container_width=True)
    with c3: st.link_button('View Kaggle source', SOURCE, use_container_width=True)

elif page == 'Dataset & Splits':
    hero('Research data', 'Dataset & Stratified Splits',
         'A reproducible video-level split. All source videos stay in Google Drive; GitHub stores metadata only.')
    c1,c2,c3=st.columns(3)
    for column,partition,desc in zip((c1,c2,c3),SPLIT_NAMES,('Model fitting','Model selection','Held-out evaluation')):
        with column: card(partition.upper(),str(summary['split_counts'][partition]),desc)
    st.subheader('Class distribution')
    data = []
    for label, splits in summary['by_class'].items():
        data.append({'Activity':display_label(label),'Train':splits['train'],
                     'Validation':splits['val'],'Test':splits['test']})
    by = pd.DataFrame(data)
    import plotly.express as px
    melted = by.melt(id_vars='Activity', var_name='Split', value_name='Videos')
    fig = px.bar(melted, x='Activity',y='Videos',color='Split',barmode='stack',
                 color_discrete_map={'Train':'#20b9a8','Validation':'#f3b45d','Test':'#689ce8'})
    render_plot(fig)
    st.dataframe(by, use_container_width=True, hide_index=True)
    st.subheader('Browse metadata')
    col_a,col_b=st.columns(2)
    with col_a: selected = st.selectbox('Activity', ['All']+[display_label(c) for c in LABELS])
    with col_b: split = st.selectbox('Partition',['All','train','val','test'])
    filtered=frame.copy()
    if selected!='All':filtered=filtered[filtered.activity.apply(display_label)==selected]
    if split!='All':filtered=filtered[filtered.split==split]
    st.dataframe(filtered.rename(columns={'video_path':'Video path','activity':'Class','bytes':'Bytes','split':'Split'}),
                 use_container_width=True,hide_index=True,height=350)
    st.caption('A random video split may include the same participant or recording environment across sets. Actor-level independent evaluation requires actor/session identifiers, which are not included in the current manifest.')

elif page == 'Predict Activity':
    hero('Live inference', 'Predict an Activity',
         'Upload an MP4 or other supported video. The existing CNN–LSTM checkpoint is loaded automatically when accessible.')
    if model_path:
        st.info(f'Loaded: **{model_path.name}**. This is a versioned training checkpoint; final test accuracy is not available. Predictions are experimental.')
    else:
        st.warning('No verified checkpoint could be loaded. Use an appropriately trained checkpoint and its matching class mapping.')
    with st.expander('Advanced: optional custom checkpoint and class mapping', expanded=False):
        supplied_model=st.file_uploader('Optional trained checkpoint (.keras)',type=['keras'],help='Keep model files private when necessary. Do not upload an untrained architecture.')
        supplied_classes=st.file_uploader('Optional classes.json from the same training run',type=['json'])
        if model_path and not supplied_model:st.success('Verified saved model is ready for inference; no model upload needed.')
        elif supplied_model:st.success('Uploaded model file selected; inference will load it on demand.')
        else:st.warning('No trained checkpoint is available yet.')
    video=st.file_uploader('Upload a video (.mp4, .avi, .mov, .mkv, .webm)',type=['mp4','avi','mov','mkv','webm'],key='inference_video')
    if video:st.caption(f'Selected: {video.name} ({len(video.getbuffer())/1_000_000:.1f} MB)')
    can_predict=bool(video and (supplied_model or model_path) and (supplied_classes or labels_path))
    if st.button('Analyze video and predict activity',type='primary',disabled=not can_predict):
        with st.spinner('Decoding video, sampling frames, and evaluating checkpoint...'):
            with tempfile.TemporaryDirectory(prefix='har_streamlit_') as temp:
                tmp=Path(temp)
                try:
                    max_video=150_000_000
                    if video.size>max_video:raise ValueError('Upload must be at most 150 MB')
                    incoming=tmp/Path(video.name).name
                    incoming.write_bytes(video.getvalue())
                    if supplied_classes:
                        classes=read_json_classes(supplied_classes.getvalue())
                    else:
                        classes=load_classes(labels_path)
                    if supplied_model:
                        checkpoint=tmp/'checkpoint.keras'
                        checkpoint.write_bytes(supplied_model.getvalue())
                        model=load_checkpoint(str(checkpoint),checkpoint.stat().st_mtime_ns)
                    else:
                        model=load_checkpoint(str(model_path),model_path.stat().st_mtime_ns)
                    ishape=model.input_shape
                    if isinstance(ishape,list):raise ValueError('Only single-input CNN–LSTM checkpoints are supported')
                    if len(ishape)!=5 or ishape[-1]!=3:raise ValueError(f'Unexpected checkpoint input shape: {ishape}')
                    if not (isinstance(model.output_shape,tuple) and model.output_shape[-1]==len(classes)):
                        raise ValueError('The class list does not match checkpoint output units')
                    t=ishape[1] or 20
                    h=ishape[2] or 64
                    w=ishape[3] or 64
                    if h!=w or t>100 or h>256:raise ValueError('Unsupported checkpoint shape: '+str(ishape))
                    import time
                    started_at=time.perf_counter()
                    info=validate_video(incoming,cache_dir=tmp/'cache')
                    x=sample_video(incoming,seq_length=int(t),image_size=int(h),cache_dir=tmp/'cache')
                    prediction=np.asarray(model.predict(x[None,...],verbose=0)[0],dtype=float)
                    elapsed=time.perf_counter()-started_at
                    if not np.isfinite(prediction).all() or (prediction < 0).any() or not np.isclose(prediction.sum(),1.0,atol=0.01):
                        raise ValueError('Checkpoint produced invalid class probabilities')
                    winner=int(prediction.argmax())
                    scores=probability_dataframe(prediction,classes)
                    st.success(f'Predicted activity: **{display_label(classes[winner])}**')
                    st.metric('Model probability (not calibrated confidence)',f'{prediction[winner]:.1%}')
                    import plotly.express as px
                    chart=px.bar(scores.sort_values('Probability'),x='Probability',y='Activity',orientation='h',
                                 range_x=[0,1],color_discrete_sequence=['#22c6aa'])
                    render_plot(chart)
                    c1,c2,c3=st.columns(3)
                    with c1:st.caption(f'Reported video frames: {info["frame_count_reported"]}')
                    with c2:st.caption(f'Preprocessing: {t} × {h} × {w} RGB')
                    with c3:st.caption(f'Processing time: {elapsed:.2f} seconds (one video)')
                    if info.get('was_zip'):
                        st.caption('The uploaded file contains a ZIP-wrapped video; displaying sampled frames instead of an undecodable raw player.')
                    else:
                        st.video(video)
                    st.caption('Three of the sampled RGB frames:')
                    st.image([np.asarray(x[i]*255,dtype=np.uint8) for i in [0,len(x)//2,-1]],width=150)
                    st.download_button('Download prediction CSV',scores.to_csv(index=False),file_name='har_prediction.csv',mime='text/csv')
                except Exception as ex:
                    st.error(f'Prediction could not complete: {ex}')
                    st.caption('Check that the model is trained and the class mapping is from the same training run. Some videos are ZIP archives disguised with .mp4 extensions; the app attempts to unwrap them.')
    if not can_predict:
        st.caption('A verified checkpoint and matching class mapping are required; otherwise no predictions are shown.')

elif page == 'Results':
    hero('Scientific evidence', 'Evaluation Results',
         'Display only measurements produced by actual CNN–LSTM training and held-out evaluation.')
    if not metrics:
        st.warning('No seven-class test_metrics.json has been published. Train in Colab, evaluate the withheld test split once, then place outputs in `outputs/` (or `artifacts/`) to populate this page.')
    else:
        c1,c2,c3=st.columns(3)
        values=[('Test accuracy',metrics.get('test_accuracy')),
                ('Macro F1',metrics.get('macro avg',{}).get('f1-score')),
                ('Weighted F1',metrics.get('weighted avg',{}).get('f1-score'))]
        for col,(name,val) in zip((c1,c2,c3),values):
            with col:st.metric(name,'—' if val is None else f'{float(val):.2%}')
        metrics_rows=[]
        for name,val in metrics.items():
            if isinstance(val,dict) and 'f1-score' in val:
                metrics_rows.append({'Class':display_label(name),'Precision':val.get('precision'),
                                    'Recall':val.get('recall'),'F1':val.get('f1-score'),'Support':val.get('support')})
        if metrics_rows:st.dataframe(pd.DataFrame(metrics_rows),use_container_width=True,hide_index=True)
    history=read_optional_csv(result_files['history'])
    if history:
        st.subheader('Training history')
        hist=pd.DataFrame(history)
        for c in hist.columns:hist[c]=pd.to_numeric(hist[c],errors='coerce')
        if 'epoch' in hist.columns:hist['epoch']=hist['epoch']+1
        if {'accuracy','val_accuracy'} <= set(hist.columns):
            st.line_chart(hist.set_index('epoch' if 'epoch' in hist.columns else hist.index)[['accuracy','val_accuracy']])
        if {'loss','val_loss'} <= set(hist.columns):
            st.line_chart(hist.set_index('epoch' if 'epoch' in hist.columns else hist.index)[['loss','val_loss']])
    else:st.caption('Training curves become available when `history.csv` is published.')
    matrix_classes=None
    if metadata and isinstance(metadata.get('classes'),list):matrix_classes=metadata['classes']
    elif model_path and labels_path:matrix_classes=load_classes(labels_path)
    if matrix_classes:
        try:
            confusion=matrix_from_csv(result_files['matrix'],matrix_classes)
            if confusion is not None:
                import plotly.express as px
                st.subheader('Confusion matrix')
                fig=px.imshow(confusion.to_numpy(),x=[display_label(c) for c in matrix_classes],
                              y=[display_label(c) for c in matrix_classes],text_auto=True,aspect='auto',
                              color_continuous_scale='Teal')
                fig.update_layout(xaxis_title='Predicted',yaxis_title='True')
                render_plot(fig)
        except Exception as error:st.warning(f'Matrix unavailable: {error}')
    st.caption('The earlier 15.42% result in the project PDF belongs to a different UCF101 experiment and is not used as a metric here.')

elif page == 'Compare Models':
    hero('Architecture benchmark', 'Compare CNN–LSTM Models',
         'Compare four designed architectures; real accuracy values appear only after training and evaluation.')
    model_df=pd.DataFrame(MODEL_VARIANTS)
    st.dataframe(model_df.rename(columns={'model':'Model','cnn':'Spatial encoder','embedding':'Frame embedding',
                                          'temporal':'Sequence model','status':'Training status'}).drop(columns='key'),
                 use_container_width=True,hide_index=True)
    comparison=read_optional_csv(result_files['comparison'])
    if comparison:
        result=pd.DataFrame(comparison)
        st.subheader('Executed experiments')
        st.dataframe(result,use_container_width=True,hide_index=True)
        if {'model','test_accuracy'}<=set(result.columns):
            import plotly.express as px
            result['test_accuracy']=pd.to_numeric(result['test_accuracy'],errors='coerce')
            render_plot(px.bar(result.dropna(subset=['test_accuracy']),x='model',y='test_accuracy',
                               color_discrete_sequence=['#22c6aa']))
    else:
        st.info('No `model_comparison.csv` was found. This section does not assign made-up performance scores to the variants.')
    a,b=st.columns(2)
    with a:
        st.markdown('#### Scratch CNN–LSTM')
        st.write('Trains convolutional filters from the project data. Variants differ in CNN depth, embedding type and LSTM size.')
    with b:
        st.markdown('#### Transfer-learning CNN–LSTM')
        st.write('Uses a frozen ImageNet-pretrained MobileNetV2 encoder and a trainable temporal LSTM. It still requires training on the target activity classes.')

elif page == 'Sensitivity & Diagnostics':
    hero('Interactive exploration', 'Preprocessing Sensitivity',
         'Understand how sampling length and frame resolution affect input memory and model cost. These are analytical estimates, not measured accuracy results.')
    a,b,c=st.columns(3)
    with a:length=st.slider('Frames per video',8,64,20,4)
    with b:size=st.select_slider('Frame edge (pixels)',options=[32,48,64,96,128,160,224],value=64)
    with c:batch=st.slider('Batch size',1,32,8,1)
    count=length*size*size*3
    memory= count*4*batch/(1024**2)
    a,b,c=st.columns(3)
    with a:st.metric('Input tensor',f'{length} × {size} × {size} × 3')
    with b:st.metric('Float32 batch memory',f'{memory:.2f} MiB')
    with c:st.metric('Input scalar values',f'{count:,}')
    st.caption('Input memory only: excludes CNN activations, optimizer states, model weights and video decoder buffers. Not a GPU usage prediction.')
    st.latex(r'M_{\mathrm{input}} = \frac{B\,T\,H\,W\,C\cdot 4}{2^{20}}\quad\text{MiB for float32}')
    st.markdown('#### Diagnostics checklist')
    for title,detail in [
        ('Frame extraction','A ZIP file saved as .mp4 must be unpacked before OpenCV reads it.'),
        ('Class leakage','All three partitions must contain different video paths; actor/session overlap is unknown.'),
        ('Validation','Choose the model using validation performance, not the test set.'),
        ('Checkpoint provenance','A matching classes.json is mandatory for trustworthy output labels.'),
        ('Latency','Run inference with your actual deployed hardware; no latency benchmark is claimed here.')]:
        with st.expander(title):st.write(detail)

elif page == 'Concepts & Mathematics':
    hero('First principles', 'Concepts & Mathematics',
         'The mathematical operations behind 20-frame CNN–LSTM human activity recognition, from pixel tensors to class probabilities.')
    st.subheader('1. Uniform frame sampling')
    st.latex(r't_i=\left\lfloor\frac{i(N-1)}{T-1}\right\rfloor,\qquad i=0,\ldots,T-1')
    st.write('From a video of N frames, select T frames distributed across the clip. Here T = 20 by default. Resize each frame to 64 × 64 RGB and scale pixel values to [0,1].')
    st.latex(r'X\in[0,1]^{T\times H\times W\times 3}')
    st.subheader('2. CNN spatial feature extraction')
    st.latex(r'z_{i,j,k}^{(\ell)}=\sigma\!\left(\sum_{u,v,c}W_{u,v,c,k}^{(\ell)}a_{i+u,j+v,c}^{(\ell-1)}+b_k^{(\ell)}\right)')
    st.write('A shared convolutional encoder is applied to each frame with TimeDistributed. The output is pooled or flattened into a feature vector fₜ.')
    st.subheader('3. LSTM temporal learning')
    st.latex(r'\begin{aligned}i_t&=\sigma(W_i[f_t,h_{t-1}]+b_i)\\f_t^{(g)}&=\sigma(W_f[f_t,h_{t-1}]+b_f)\\o_t&=\sigma(W_o[f_t,h_{t-1}]+b_o)\\\tilde c_t&=\tanh(W_c[f_t,h_{t-1}]+b_c)\\c_t&=f_t^{(g)}\odot c_{t-1}+i_t\odot\tilde c_t\\h_t&=o_t\odot\tanh(c_t)\end{aligned}')
    st.write('The forget, input, and output gates select useful temporal context. The notation fₜ^(g) denotes the forget gate; fₜ in the CNN section denotes the frame feature.')
    st.subheader('4. Softmax classification and loss')
    st.latex(r'p(y=k\mid X)=\frac{\exp((Wh_T+b)_k)}{\sum_{j=1}^{K}\exp((Wh_T+b)_j)}')
    st.latex(r'\mathcal L_{\mathrm{CE}}=-\frac1M\sum_{i=1}^{M}\log p(y_i\mid X_i)')
    st.write('For this Kaggle subset, K = 7 activity categories. Adam optimizes sparse categorical cross-entropy; dropout and early stopping support regularization.')
    st.subheader('5. Evaluation')
    st.latex(r'\mathrm{Precision}=\frac{TP}{TP+FP},\quad\mathrm{Recall}=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}')
    st.write('Compute accuracy, macro F1 and a confusion matrix on a held-out test set only after selecting the model with validation data.')
    st.link_button('Read detailed methodology',GITHUB+'/blob/main/docs/METHODOLOGY.md')

elif page == 'Downloads':
    hero('Project materials', 'Downloads & Reproducibility',
         'Export the observed data split, summary and available experiment artifacts.')
    st.download_button('Download stratified 70/15/15 split (CSV)',dataframe_csv_bytes(rows),
                       'train_val_test_splits_70_15_15.csv','text/csv',use_container_width=True)
    snapshot=read_optional_json(DATA_JSON)
    if snapshot:
        st.download_button('Download dataset summary (JSON)',json.dumps(snapshot,indent=2),
                           'dataset_summary_70_15_15.json','application/json',use_container_width=True)
    for name,path in result_files.items():
        if path.is_file():
            mime='application/json' if path.suffix=='.json' else 'text/csv'
            st.download_button(f'Download {path.name}',path.read_bytes(),file_name=path.name,mime=mime,
                               use_container_width=True,key='download_'+name)
    st.markdown('#### External project resources')
    st.link_button('GitHub source and README',GITHUB,use_container_width=True)
    st.link_button('Complete training Colab',COLAB,use_container_width=True)
    st.link_button('Video dataset in Drive',DRIVE,use_container_width=True)
    st.caption('Full dataset videos are not included in the repository or exported by this page.')

else:
    hero('People & provenance', 'About the Project',
         'Human Activity Recognition from Video Using Convolutional Neural Networks and Long Short-Term Memory Networks.')
    st.write('**Student:** Raj Rishav  \n**Department:** Electronics and Electrical Engineering  \n**Institute:** Indian Institute of Technology Guwahati')
    st.write('**Dataset:** Kaggle Human Activity Recognition – Video Dataset (seven classes; 132-video subset).')
    st.write('**Research objective:** Model spatial visual patterns with a CNN and temporal dependencies with an LSTM, evaluate across seven activities, and compare scratch-trained and transfer-learning designs.')
    st.warning('The original PDF uses UCF101 (101 categories) and an 80/20 split. This implementation intentionally uses the newer user-selected seven-class Kaggle dataset and a separate 70/15/15 split. Previously published numbers are not transferred to the new task.')
    st.link_button('View repository',GITHUB)

st.divider()
st.caption('Raj Rishav • IIT Guwahati • Human Activity Recognition | CNN + LSTM  ·  Real dataset metadata, evidence-first results')
