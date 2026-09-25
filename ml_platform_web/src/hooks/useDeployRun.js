import { useState } from 'react'
import { message } from 'antd'
import { modelingTaskApi } from '../services/api'

export const deploymentNameError = (name) => name?.trim() ? null : '请填写部署名称'

/**
 * useDeployRun — shared deploy guardrails for a modeling task's runs.
 *
 * Deploys a run's model via the V3 bridge (`modelingTaskApi.deployRun`). Owns
 * deployment-name validation, response state, and error handling so both
 * DeployStep and the ModelComparison row action behave identically. Callers
 * should still gate the trigger UI on `status === 'SUCCESS' && domain_task_id`.
 *
 * @param {{id: string, name: string}} task
 * @returns {{deploying: boolean, deployment: object|null, error: string|null, deploy: (runId: string, opts?: {name?: string, description?: string}) => Promise<object|null>}}
 */
export function useDeployRun(task) {
  const [deploying, setDeploying] = useState(false)
  const [deployment, setDeployment] = useState(null)
  const [error, setError] = useState(null)

  const deploy = async (runId, { name, description } = {}) => {
    if (!runId) { message.warning('请先选择一个成功的 Run'); return null }
    const nameError = deploymentNameError(name)
    if (nameError) { message.warning(nameError); return null }
    setDeploying(true)
    setError(null)
    try {
      const resp = await modelingTaskApi.deployRun(task.id, runId, {
        name: name.trim(),
        // A caller-written note when there is one; the generic sentence is
        // only a fallback so the field is never empty in the deployment list.
        description: description?.trim() || `来自建模任务 ${task.name} 的模型`,
      })
      setDeployment(resp)
      message.success('部署成功，模型已上线')
      return resp
    } catch (err) {
      const detail = err?.response?.data?.detail || '部署失败'
      setError(detail)
      message.error(detail)
      return null
    } finally {
      setDeploying(false)
    }
  }

  const reset = () => { setDeployment(null); setError(null) }

  return { deploying, deployment, error, deploy, reset }
}
