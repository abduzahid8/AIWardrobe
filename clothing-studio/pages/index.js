import { useState, useRef, useCallback } from "react";

const STEPS = {
  UPLOAD: "upload", ANALYZING: "analyzing",
  GENERATE: "generate", GENERATING: "generating",
  RESULT: "result", ERROR: "error"
};

function toBase64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result.split(",")[1]);
    r.onerror = () => rej(new Error("Read failed"));
    r.readAsDataURL(file);
  });
}

export default function ClothingStudio() {
  const [step, setStep] = useState(STEPS.UPLOAD);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [file, setFile] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [resultUrl, setResultUrl] = useState(null);
  const [error, setError] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [editedPrompt, setEditedPrompt] = useState("");
  const [imgLoaded, setImgLoaded] = useState(false);
  const [statusMsg, setStatusMsg] = useState("");
  const fileRef = useRef();

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setStep(STEPS.UPLOAD);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFile(e.dataTransfer.files[0]);
  }, [handleFile]);

  const runAnalysis = async () => {
    if (!file) return;
    setStep(STEPS.ANALYZING);
    setError("");
    try {
      const base64 = await toBase64(file);
      const res = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: base64, mediaType: file.type })
      });
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setAnalysis(data);
      setEditedPrompt(data.generationPrompt);
      setStep(STEPS.GENERATE);
    } catch (e) {
      setError("Analysis failed: " + e.message);
      setStep(STEPS.ERROR);
    }
  };

  const runGeneration = async () => {
    setStep(STEPS.GENERATING);
    setImgLoaded(false);
    setError("");
    setStatusMsg("Sending to Flux...");
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: editedPrompt })
      });
      setStatusMsg("Generating your product shot...");
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setResultUrl(data.image);
      setStep(STEPS.RESULT);
    } catch (e) {
      setError("Generation failed: " + e.message);
      setStep(STEPS.ERROR);
    }
  };

  const reset = () => {
    setStep(STEPS.UPLOAD); setPreviewUrl(null); setFile(null);
    setAnalysis(null); setResultUrl(null);
    setError(""); setEditedPrompt(""); setImgLoaded(false); setStatusMsg("");
  };

  const downloadImage = () => {
    const a = document.createElement("a");
    a.href = resultUrl;
    a.download = "product-shot.webp";
    a.click();
  };

  return (
    <>
      <style dangerouslySetInnerHTML={{
        __html: `
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #f8f6f2; }
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,300;0,400;1,300&family=Jost:wght@300;400&display=swap');
        @keyframes pulse { 0%,80%,100%{opacity:0.15} 40%{opacity:1} }
        @keyframes shimmer { 0%{transform:translateX(-100%)} 100%{transform:translateX(250%)} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(16px)} to{opacity:1;transform:translateY(0)} }
        .fade-up { animation: fadeUp 0.5s ease forwards; }
        .btn-primary {
          background: #1a1a1a; color: #f8f6f2; border: none;
          padding: 15px 44px; cursor: pointer;
          letter-spacing: 0.25em; font-size: 11px;
          font-family: 'Jost', sans-serif; font-weight: 300;
          transition: background 0.2s;
        }
        .btn-primary:hover { background: #333; }
        .btn-secondary {
          background: none; border: 1px solid #ccc; color: #666;
          padding: 13px 28px; cursor: pointer;
          letter-spacing: 0.2em; font-size: 11px;
          font-family: 'Jost', sans-serif; font-weight: 300;
          transition: all 0.2s;
        }
        .btn-secondary:hover { border-color: #1a1a1a; color: #1a1a1a; }
        .upload-zone {
          border: 1px dashed #ccc;
          background: #fff;
          padding: 72px 48px;
          text-align: center;
          cursor: pointer;
          transition: all 0.3s;
        }
        .upload-zone:hover, .upload-zone.drag { border-color: #1a1a1a; background: #faf9f7; }
        .card { background: #fff; border: 1px solid #ece9e3; padding: 20px 24px; }
        .label { font-size: 9px; letter-spacing: 0.45em; color: #aaa; font-family: 'Jost', sans-serif; font-weight: 300; margin-bottom: 6px; }
        .value { font-size: 13px; color: #1a1a1a; font-family: 'Jost', sans-serif; font-weight: 300; line-height: 1.6; }
        .step-label { font-size: 9px; letter-spacing: 0.45em; color: #bbb; font-family: 'Jost', sans-serif; margin-bottom: 12px; }
        .step-label span { color: #c8a96e; }
      ` }} />

      <div style={{ minHeight: "100vh", background: "#f8f6f2", fontFamily: "'Jost', sans-serif" }}>

        {/* Header */}
        <header style={{
          borderBottom: "1px solid #e8e4dc",
          padding: "20px 56px",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          background: "#fff"
        }}>
          <div>
            <div style={{ fontSize: "9px", letterSpacing: "0.5em", color: "#bbb", marginBottom: "2px", fontWeight: 300 }}>POWERED BY AI</div>
            <div style={{ fontSize: "18px", letterSpacing: "0.3em", color: "#1a1a1a", fontFamily: "'Playfair Display', serif", fontWeight: 300 }}>
              ATELIER STUDIO
            </div>
          </div>
          <div style={{ fontSize: "9px", letterSpacing: "0.4em", color: "#ccc" }}>E-COMMERCE PHOTOGRAPHY</div>
          {step !== STEPS.UPLOAD
            ? <button className="btn-secondary" onClick={reset} style={{ padding: "7px 16px" }}>NEW IMAGE</button>
            : <div style={{ width: "96px" }} />
          }
        </header>

        <main style={{ padding: "64px 56px", maxWidth: "1000px", margin: "0 auto" }}>

          {/* UPLOAD */}
          {step === STEPS.UPLOAD && (
            <div className="fade-up" style={{ display: "flex", flexDirection: "column", gap: "44px" }}>
              <div>
                <div className="step-label"><span>01</span> — UPLOAD GARMENT</div>
                <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: "42px", fontWeight: 300, lineHeight: 1.25, color: "#1a1a1a" }}>
                  From any photo<br /><em>to product shot</em>
                </h1>
                <p style={{ marginTop: "16px", color: "#888", fontSize: "14px", lineHeight: 1.9, maxWidth: "520px", fontWeight: 300 }}>
                  Upload clothing from a hanger, store shelf, or floor. Our AI removes the background noise and recreates it as a pristine white-background catalog image.
                </p>
              </div>

              <div
                className={`upload-zone${dragOver ? " drag" : ""}`}
                onClick={() => fileRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
              >
                {previewUrl ? (
                  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "16px" }}>
                    <img src={previewUrl} alt="Preview" style={{ maxHeight: "300px", maxWidth: "100%", objectFit: "contain" }} />
                    <div style={{ fontSize: "11px", color: "#c8a96e", letterSpacing: "0.2em" }}>{file?.name}</div>
                    <div style={{ fontSize: "10px", color: "#bbb", letterSpacing: "0.15em" }}>CLICK TO CHANGE</div>
                  </div>
                ) : (
                  <div>
                    <div style={{ fontSize: "32px", color: "#ddd", marginBottom: "16px" }}>+</div>
                    <div style={{ fontSize: "11px", letterSpacing: "0.4em", color: "#bbb", marginBottom: "8px" }}>DROP PHOTO HERE</div>
                    <div style={{ fontSize: "11px", color: "#ddd", letterSpacing: "0.1em" }}>or click to browse · JPG, PNG, WEBP</div>
                  </div>
                )}
                <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={e => handleFile(e.target.files[0])} />
              </div>

              {previewUrl && (
                <div>
                  <button className="btn-primary" onClick={runAnalysis}>ANALYZE GARMENT →</button>
                </div>
              )}
            </div>
          )}

          {/* ANALYZING */}
          {step === STEPS.ANALYZING && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "420px", gap: "28px" }}>
              <div style={{ fontSize: "10px", letterSpacing: "0.45em", color: "#bbb" }}>ANALYZING GARMENT</div>
              <div style={{ display: "flex", gap: "10px" }}>
                {[0, 1, 2].map(i => (
                  <div key={i} style={{ width: "7px", height: "7px", background: "#c8a96e", borderRadius: "50%", animation: `pulse 1.4s ease-in-out ${i * 0.22}s infinite` }} />
                ))}
              </div>
              <div style={{ fontSize: "12px", color: "#ccc", letterSpacing: "0.2em" }}>Gemini is reading your photo...</div>
            </div>
          )}

          {/* GENERATE */}
          {step === STEPS.GENERATE && analysis && (
            <div className="fade-up" style={{ display: "flex", flexDirection: "column", gap: "36px" }}>
              <div>
                <div className="step-label"><span>02</span> — REVIEW ANALYSIS</div>
                <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: "32px", fontWeight: 300, color: "#1a1a1a" }}>
                  Garment Identified
                </h2>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: "1px", background: "#e8e4dc" }}>
                {[
                  ["TYPE", analysis.garmentType],
                  ["COLOR", analysis.color],
                  ["MATERIAL", analysis.material || "—"],
                  ["CATEGORY", analysis.category]
                ].map(([l, v]) => (
                  <div key={l} className="card" style={{ borderRadius: 0, border: "none" }}>
                    <div className="label">{l}</div>
                    <div className="value">{v}</div>
                  </div>
                ))}
              </div>

              <div className="card">
                <div className="label">DETAILS</div>
                <div className="value" style={{ lineHeight: 1.9 }}>{analysis.details}</div>
              </div>

              <div>
                <div className="label" style={{ marginBottom: "10px" }}>GENERATION PROMPT <span style={{ color: "#ddd" }}>— EDIT TO REFINE</span></div>
                <textarea
                  value={editedPrompt}
                  onChange={e => setEditedPrompt(e.target.value)}
                  rows={5}
                  style={{
                    width: "100%", background: "#fff", border: "1px solid #e0dcd4",
                    color: "#555", padding: "14px 16px", fontSize: "12px",
                    lineHeight: 1.8, fontFamily: "monospace", resize: "vertical", outline: "none"
                  }}
                />
              </div>

              <div style={{ background: "#fff", border: "1px solid #e8e4dc", borderLeft: "3px solid #c8a96e", padding: "14px 18px", display: "flex", gap: "12px" }}>
                <div style={{ fontSize: "12px", color: "#888", lineHeight: 1.7, fontWeight: 300 }}>
                  Generation runs server-side via <strong style={{ color: "#c8a96e" }}>Replicate (Flux)</strong>. Set your <code style={{ background: "#f5f3ef", padding: "1px 5px", fontSize: "11px" }}>REPLICATE_API_KEY</code> in <code style={{ background: "#f5f3ef", padding: "1px 5px", fontSize: "11px" }}>.env.local</code>.
                </div>
              </div>

              <div>
                <button className="btn-primary" onClick={runGeneration}>GENERATE PRODUCT SHOT →</button>
              </div>
            </div>
          )}

          {/* GENERATING */}
          {step === STEPS.GENERATING && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "420px", gap: "28px" }}>
              <div style={{ fontSize: "10px", letterSpacing: "0.45em", color: "#bbb" }}>GENERATING</div>
              <div style={{ width: "240px", height: "1px", background: "#e8e4dc", position: "relative", overflow: "hidden" }}>
                <div style={{ position: "absolute", left: 0, top: 0, height: "100%", width: "50%", background: "#c8a96e", animation: "shimmer 1.8s ease-in-out infinite" }} />
              </div>
              <div style={{ fontSize: "12px", color: "#bbb", letterSpacing: "0.2em" }}>{statusMsg}</div>
              <div style={{ fontSize: "11px", color: "#d5d0c8", letterSpacing: "0.1em" }}>10–30 seconds</div>
            </div>
          )}

          {/* RESULT */}
          {step === STEPS.RESULT && resultUrl && (
            <div className="fade-up" style={{ display: "flex", flexDirection: "column", gap: "40px" }}>
              <div>
                <div className="step-label"><span>03</span> — RESULT</div>
                <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: "32px", fontWeight: 300, color: "#1a1a1a" }}>
                  Product Shot Ready
                </h2>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "24px" }}>
                <div>
                  <div className="label" style={{ marginBottom: "10px" }}>ORIGINAL PHOTO</div>
                  <div style={{ background: "#f0ede8", border: "1px solid #e8e4dc", height: "420px", display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden" }}>
                    <img src={previewUrl} alt="Original" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }} />
                  </div>
                </div>
                <div>
                  <div className="label" style={{ marginBottom: "10px", color: "#c8a96e" }}>GENERATED PRODUCT SHOT ✦</div>
                  <div style={{ background: "#fff", border: "1px solid #e8e4dc", height: "420px", display: "flex", alignItems: "center", justifyContent: "center", overflow: "hidden", position: "relative" }}>
                    {!imgLoaded && (
                      <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center", background: "#f8f6f2" }}>
                        <div style={{ display: "flex", gap: "8px" }}>
                          {[0, 1, 2].map(i => <div key={i} style={{ width: "6px", height: "6px", background: "#c8a96e", borderRadius: "50%", animation: `pulse 1.4s ${i * 0.2}s infinite` }} />)}
                        </div>
                      </div>
                    )}
                    <img src={resultUrl} alt="Product shot" onLoad={() => setImgLoaded(true)}
                      style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", opacity: imgLoaded ? 1 : 0, transition: "opacity 0.6s" }} />
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
                <button className="btn-primary" onClick={downloadImage}>DOWNLOAD IMAGE ↓</button>
                <button className="btn-secondary" onClick={runGeneration}>REGENERATE</button>
                <button className="btn-secondary" onClick={() => setStep(STEPS.GENERATE)}>EDIT PROMPT</button>
                <button className="btn-secondary" onClick={reset}>NEW ITEM</button>
              </div>
            </div>
          )}

          {/* ERROR */}
          {step === STEPS.ERROR && (
            <div className="fade-up" style={{ display: "flex", flexDirection: "column", gap: "20px", maxWidth: "560px" }}>
              <div style={{ fontSize: "10px", color: "#e06060", letterSpacing: "0.4em" }}>ERROR</div>
              <div style={{ background: "#fff", border: "1px solid #f0dada", borderLeft: "3px solid #e06060", padding: "20px", fontSize: "13px", color: "#c06060", lineHeight: 1.9, fontFamily: "monospace" }}>
                {error}
              </div>
              <div>
                <button className="btn-secondary" onClick={() => setStep(analysis ? STEPS.GENERATE : STEPS.UPLOAD)}>← TRY AGAIN</button>
              </div>
            </div>
          )}

        </main>

        <footer style={{ borderTop: "1px solid #e8e4dc", padding: "18px 56px", display: "flex", justifyContent: "space-between", alignItems: "center", background: "#fff" }}>
          <div style={{ fontSize: "9px", color: "#ccc", letterSpacing: "0.3em" }}>GEMINI × FLUX · REPLICATE</div>
          <div style={{ display: "flex", gap: "20px" }}>
            {["01", "02", "03"].map((n, i) => {
              const on = (i === 0 && [STEPS.UPLOAD, STEPS.ANALYZING].includes(step)) ||
                (i === 1 && [STEPS.GENERATE, STEPS.GENERATING].includes(step)) ||
                (i === 2 && step === STEPS.RESULT);
              return <div key={n} style={{ fontSize: "9px", letterSpacing: "0.3em", color: on ? "#c8a96e" : "#ddd" }}>{n}</div>;
            })}
          </div>
        </footer>
      </div>
    </>
  );
}
