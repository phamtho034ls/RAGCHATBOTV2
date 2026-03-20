import React, { useState, useCallback } from "react";
import { Upload, X, FileUp, Folder, Loader2, CheckCircle, XCircle } from "lucide-react";
import { uploadDocx, uploadFolder } from "../api/client";

export default function UploadModal({ onClose, onDone }) {
  const [mode, setMode] = useState("file"); // "file" | "folder"

  // Single file
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);

  // Folder
  const [folderFiles, setFolderFiles] = useState([]); // File[]
  const [folderResults, setFolderResults] = useState(null); // FolderUploadResponse

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ── Single file handlers ──
  const handleFile = (f) => {
    setError("");
    if (!f.name.endsWith(".docx") && !f.name.endsWith(".doc")) {
      setError("Chỉ hỗ trợ file .doc và .docx");
      return;
    }
    setFile(f);
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, []);

  // ── Folder handler ──
  const handleFolderSelect = (e) => {
    setError("");
    setFolderResults(null);
    const allFiles = Array.from(e.target.files || []);
    const wordFiles = allFiles.filter(
      (f) => f.name.endsWith(".doc") || f.name.endsWith(".docx")
    );
    if (wordFiles.length === 0) {
      setError("Folder không chứa file .doc hoặc .docx nào");
      return;
    }
    setFolderFiles(wordFiles);
  };

  // ── Upload single file ──
  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const data = await uploadDocx(file);
      onDone(data);
    } catch (err) {
      setError(err.message || "Upload thất bại");
    } finally {
      setLoading(false);
    }
  };

  // ── Upload folder ──
  const handleUploadFolder = async () => {
    if (folderFiles.length === 0) return;
    setLoading(true);
    setError("");
    setFolderResults(null);
    try {
      const data = await uploadFolder(folderFiles);
      setFolderResults(data);
      // Notify parent with list of successful uploads
      const successItems = data.results.filter((r) => r.success);
      if (successItems.length > 0) onDone(successItems[0], data);
    } catch (err) {
      setError(err.message || "Upload folder thất bại");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="bg-surface-card border border-surface-border rounded-2xl w-full max-w-md mx-4 p-6 animate-slide-up">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-100">Tải lên tài liệu</h2>
          <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-surface-hover text-gray-400">
            <X size={18} />
          </button>
        </div>

        {/* Mode tabs */}
        <div className="flex gap-1 mb-4 bg-surface-hover rounded-xl p-1">
          <button
            onClick={() => { setMode("file"); setError(""); setFolderResults(null); }}
            className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-sm transition ${
              mode === "file" ? "bg-surface-card text-gray-100" : "text-gray-400 hover:text-gray-300"
            }`}
          >
            <FileUp size={14} /> File đơn
          </button>
          <button
            onClick={() => { setMode("folder"); setError(""); setFile(null); }}
            className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-lg text-sm transition ${
              mode === "folder" ? "bg-surface-card text-gray-100" : "text-gray-400 hover:text-gray-300"
            }`}
          >
            <Folder size={14} /> Cả folder
          </button>
        </div>

        {/* ── Single file zone ── */}
        {mode === "file" && (
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => document.getElementById("file-input").click()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors ${
              dragging ? "border-primary-400 bg-primary-600/10" : "border-surface-border hover:border-gray-500"
            }`}
          >
            <input
              id="file-input"
              type="file"
              accept=".doc,.docx"
              className="hidden"
              onChange={(e) => e.target.files[0] && handleFile(e.target.files[0])}
            />
            {file ? (
              <div className="flex flex-col items-center gap-2">
                <FileUp size={32} className="text-primary-400" />
                <p className="text-sm text-gray-300">{file.name}</p>
                <p className="text-xs text-gray-500">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Upload size={32} className="text-gray-500" />
                <p className="text-sm text-gray-400">Kéo thả hoặc nhấn để chọn file .doc / .docx</p>
              </div>
            )}
          </div>
        )}

        {/* ── Folder zone ── */}
        {mode === "folder" && !folderResults && (
          <div
            onClick={() => document.getElementById("folder-input").click()}
            className="border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors border-surface-border hover:border-gray-500"
          >
            <input
              id="folder-input"
              type="file"
              accept=".doc,.docx"
              multiple
              /* webkitdirectory lets browsers expose the folder picker */
              webkitdirectory=""
              className="hidden"
              onChange={handleFolderSelect}
            />
            {folderFiles.length > 0 ? (
              <div className="flex flex-col items-center gap-2">
                <Folder size={32} className="text-primary-400" />
                <p className="text-sm text-gray-300">
                  {folderFiles.length} file .doc/.docx được chọn
                </p>
                <p className="text-xs text-gray-500">
                  {(folderFiles.reduce((s, f) => s + f.size, 0) / 1024).toFixed(1)} KB
                </p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-2">
                <Folder size={32} className="text-gray-500" />
                <p className="text-sm text-gray-400">Nhấn để chọn folder chứa file .doc / .docx</p>
                <p className="text-xs text-gray-500">Chỉ các file .doc và .docx sẽ được xử lý</p>
              </div>
            )}
          </div>
        )}

        {/* ── Folder results ── */}
        {mode === "folder" && folderResults && (
          <div className="rounded-xl border border-surface-border overflow-hidden">
            <div className="px-4 py-2 bg-surface-hover flex items-center justify-between">
              <span className="text-xs text-gray-400">
                Thành công: <span className="text-green-400 font-medium">{folderResults.success_count}</span>
                {" / "}
                Lỗi: <span className="text-red-400 font-medium">{folderResults.fail_count}</span>
                {" / "}
                Tổng: <span className="text-gray-300 font-medium">{folderResults.total_files}</span>
              </span>
            </div>
            <ul className="max-h-48 overflow-y-auto divide-y divide-surface-border">
              {folderResults.results.map((r, i) => (
                <li key={i} className="flex items-start gap-2 px-4 py-2">
                  {r.success
                    ? <CheckCircle size={14} className="text-green-400 mt-0.5 shrink-0" />
                    : <XCircle size={14} className="text-red-400 mt-0.5 shrink-0" />}
                  <div className="min-w-0">
                    <p className="text-xs text-gray-300 truncate">{r.file_name}</p>
                    {r.success
                      ? <p className="text-xs text-gray-500">{r.total_chunks} chunks</p>
                      : <p className="text-xs text-red-400 truncate">{r.error}</p>}
                  </div>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Error */}
        {error && <p className="mt-3 text-sm text-red-400 text-center">{error}</p>}

        {/* Actions */}
        <div className="flex gap-3 mt-5">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2.5 rounded-xl border border-surface-border text-sm text-gray-400 hover:bg-surface-hover transition"
          >
            {folderResults ? "Đóng" : "Hủy"}
          </button>
          {!folderResults && (
            <button
              onClick={mode === "file" ? handleUpload : handleUploadFolder}
              disabled={loading || (mode === "file" ? !file : folderFiles.length === 0)}
              className="flex-1 px-4 py-2.5 rounded-xl bg-primary-600 text-sm text-white hover:bg-primary-700 disabled:opacity-40 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  Đang xử lý…
                </>
              ) : (
                "Tải lên & Xử lý"
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

