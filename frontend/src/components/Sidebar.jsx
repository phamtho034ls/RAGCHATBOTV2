import React, { useState, useEffect } from "react";
import {
  FileText,
  Trash2,
  Plus,
  ChevronLeft,
  ChevronRight,
  Settings,
} from "lucide-react";
import { getDatasets, deleteDataset } from "../api/client";
import UploadModal from "./UploadModal";
import TemperatureSlider from "./TemperatureSlider";
import GpuBadge from "./GpuBadge";

export default function Sidebar({
  selectedDataset,
  onSelectDataset,
  temperature,
  onTemperatureChange,
  onDatasetsChange,
}) {
  const [datasets, setDatasets] = useState([]);
  const [showUpload, setShowUpload] = useState(false);
  const [collapsed, setCollapsed] = useState(false);

  const refreshDatasets = async () => {
    try {
      const res = await getDatasets();
      const list = res.datasets || [];
      setDatasets(list);
      onDatasetsChange?.(list);
      return list;
    } catch {
      /* ignore */
      return [];
    }
  };

  useEffect(() => {
    refreshDatasets().then((list) => {
      if (!selectedDataset && list.length > 0) {
        onSelectDataset(list[0].dataset_id);
      }
    });
  }, []);

  const handleDelete = async (id) => {
    if (!confirm("Bạn có chắc muốn xóa tài liệu này?")) return;
    try {
      await deleteDataset(id);
      if (selectedDataset === id) onSelectDataset(null);
      refreshDatasets();
    } catch {
      /* ignore */
    }
  };

  const handleUploadDone = (data) => {
    setShowUpload(false);
    refreshDatasets();
    onSelectDataset(data.dataset_id);
  };

  if (collapsed) {
    return (
      <div className="flex flex-col items-center py-4 gap-3 bg-surface-card border-r border-surface-border w-12">
        <button
          onClick={() => setCollapsed(false)}
          className="p-2 rounded-lg hover:bg-surface-hover text-gray-400"
        >
          <ChevronRight size={18} />
        </button>
      </div>
    );
  }

  return (
    <>
      <aside className="w-72 bg-surface-card border-r border-surface-border flex flex-col h-full">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-surface-border">
          <h1 className="text-base font-semibold text-gray-100 flex items-center gap-2">
            <span className="text-xl">🤖</span> RAG Chat
          </h1>
          <button
            onClick={() => setCollapsed(true)}
            className="p-1.5 rounded-lg hover:bg-surface-hover text-gray-400"
          >
            <ChevronLeft size={16} />
          </button>
        </div>

        {/* Upload button */}
        <div className="px-3 pt-3">
          <button
            onClick={() => setShowUpload(true)}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-xl border border-dashed border-surface-border hover:border-primary-400 hover:bg-surface-hover text-sm text-gray-300 transition-colors"
          >
            <Plus size={16} />
            Tải lên tài liệu (.doc, .docx)
          </button>
        </div>

        {/* Dataset list */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1">
          <p className="text-xs text-gray-500 uppercase tracking-wider px-1 py-2">
            Tài liệu
          </p>
          {datasets.length === 0 && (
            <p className="text-xs text-gray-600 px-1">
              Chưa có tài liệu nào. Hãy tải lên file .doc / .docx để bắt đầu.
            </p>
          )}
          {datasets.map((ds) => (
            <div
              key={ds.dataset_id}
              onClick={() => onSelectDataset(ds.dataset_id)}
              className={`group flex items-center gap-2 px-3 py-2 rounded-xl cursor-pointer text-sm transition-colors ${
                selectedDataset === ds.dataset_id
                  ? "bg-primary-600/15 text-primary-400 border border-primary-600/30"
                  : "text-gray-400 hover:bg-surface-hover hover:text-gray-200 border border-transparent"
              }`}
            >
              <FileText size={15} className="flex-shrink-0" />
              <span className="flex-1 truncate">{ds.file_name}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(ds.dataset_id);
                }}
                className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-500/20 text-gray-500 hover:text-red-400 transition"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>

        {/* Settings */}
        <div className="border-t border-surface-border px-4 py-3 space-y-3">
          <div className="flex items-center gap-2 text-xs text-gray-500 uppercase tracking-wider">
            <Settings size={13} />
            Cài đặt
          </div>
          <TemperatureSlider value={temperature} onChange={onTemperatureChange} />
          <GpuBadge />
        </div>
      </aside>

      {showUpload && (
        <UploadModal onClose={() => setShowUpload(false)} onDone={handleUploadDone} />
      )}
    </>
  );
}
