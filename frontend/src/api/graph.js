import service from './index'

/**
 * 生成本体（上传文档和模拟需求）
 * @param {Object} data - 包含files, simulation_requirement, project_name等
 * @returns {Promise}
 */
export function generateOntology(formData) {
  return service({
    url: '/api/graph/ontology/generate',
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

/**
 * 构建图谱
 * @param {Object} data - 包含project_id, graph_name等
 * @returns {Promise}
 */
export function buildGraph(data) {
  return service({
    url: '/api/graph/build',
    method: 'post',
    data
  })
}

/**
 * 查询任务状态
 * @param {String} taskId - 任务ID
 * @returns {Promise}
 */
export function getTaskStatus(taskId) {
  return service({
    url: `/api/graph/task/${taskId}`,
    method: 'get'
  })
}

/**
 * 获取图谱数据
 * Zep's FREE plan allows only a few requests per minute. When the backend answers "zep_rate_limited", wait the time
 * it names and ask again (up to 4 times); `onWait(seconds)` lets the page say so instead of showing an empty graph.
 * @param {String} graphId - 图谱ID
 * @param {Function} [onWait] - called with the seconds about to be waited
 * @returns {Promise}
 */
export async function getGraphData(graphId, onWait) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await service({
        url: `/api/graph/data/${graphId}`,
        method: 'get'
      })
    } catch (error) {
      const body = error.response?.data
      if (error.response?.status !== 429 || body?.error_code !== 'zep_rate_limited' || attempt >= 4) {
        throw error
      }
      const seconds = Math.min(Math.max(Number(body.retry_after_seconds) || 30, 1), 90)
      if (onWait) onWait(seconds)
      await new Promise(resolve => setTimeout(resolve, seconds * 1000))
    }
  }
}

/**
 * 获取项目信息
 * @param {String} projectId - 项目ID
 * @returns {Promise}
 */
export function getProject(projectId) {
  return service({
    url: `/api/graph/project/${projectId}`,
    method: 'get'
  })
}
