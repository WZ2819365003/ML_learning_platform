import { describe, expect, it } from 'vitest'
import {
  inferTimeColumn,
  inferValueColumn,
  latestSuccessfulTask,
  missingTsFields,
  payloadFromPreview,
  payloadFromTask,
} from './tsTestPayload'

function preview(columns) {
  const columns_info = {}
  const statistics = {}
  for (const [name, dtype, unique] of columns) {
    columns_info[name] = { dtype }
    statistics[name] = { unique_count: unique }
  }
  return { columns_info, statistics, row_count: 87312 }
}

// 线上那份 电力负荷预测数据.csv 的真实列（节选，含全部会干扰判断的派生特征）。
const LOAD_PREVIEW = preview([
  ['hour_of_day', 'float64', 48],
  ['day_of_week', 'int64', 3],
  ['is_weekend', 'int64', 2],
  ['month', 'int64', 1],
  ['days_since_start', 'float64', 100],
  ['hour_sin', 'float64', 41],
  ['hour_cos', 'float64', 40],
  ['dry_bulb_temp', 'float64', 65],
  ['dry_bulb_temp_sq', 'float64', 65],
  ['load_lag_1', 'float64', 100],
  ['load_lag_48', 'float64', 100],
  ['load_roll_mean_48', 'float64', 100],
  ['load_roll_std_48', 'float64', 100],
  ['load', 'float64', 100],
])

const PV_PREVIEW = preview([
  ['days_since_start', 'float64', 100],
  ['hour_of_day', 'float64', 48],
  ['irradiance', 'float64', 95],
  ['module_temp', 'float64', 88],
  ['ac_power', 'float64', 99],
])

describe('inferTimeColumn', () => {
  it('单调列胜过周期列——days_since_start 画出来的轴才有意义', () => {
    expect(inferTimeColumn(LOAD_PREVIEW)).toBe('days_since_start')
  })

  it('真的有 datetime 列时优先用它', () => {
    const p = preview([['days_since_start', 'float64', 100], ['ts', 'datetime64[ns]', 100]])
    expect(inferTimeColumn(p)).toBe('ts')
  })

  it('一个时间线索都没有就返回 null，而不是硬凑一个', () => {
    expect(inferTimeColumn(preview([['a', 'float64', 10], ['b', 'float64', 10]]))).toBeNull()
  })
})

describe('inferValueColumn', () => {
  it('选 load，不选 load_lag_1 / load_roll_mean_48', () => {
    expect(inferValueColumn(LOAD_PREVIEW, 'days_since_start')).toBe('load')
  })

  it('光伏那份选 ac_power', () => {
    expect(inferValueColumn(PV_PREVIEW, 'days_since_start')).toBe('ac_power')
  })

  it('绝不会把时间列当成目标值', () => {
    expect(inferValueColumn(LOAD_PREVIEW, 'hour_of_day')).not.toBe('hour_of_day')
  })

  it('没有名字线索时退回抽样去重值最多的数值列，跳过派生特征', () => {
    const p = preview([
      ['flag', 'int64', 2],
      ['x_sin', 'float64', 99],
      ['reading', 'float64', 90],
    ])
    expect(inferValueColumn(p, null)).toBe('reading')
  })

  it('没有数值列就返回 null', () => {
    expect(inferValueColumn(preview([['name', 'object', 10]]), null)).toBeNull()
  })
})

describe('payloadFromPreview', () => {
  it('给出可以直接发送的请求体', () => {
    expect(payloadFromPreview({ id: 'ds-1' }, LOAD_PREVIEW)).toEqual({
      dataset_id: 'ds-1',
      value_column: 'load',
      time_column: 'days_since_start',
      horizon: 24,
      frequency: 'D',
    })
  })

  it('推不出时间列时留 null——后端签名是 str | None，不影响预测', () => {
    const p = preview([['reading', 'float64', 90]])
    expect(payloadFromPreview({ id: 'ds-2' }, p)).toEqual({
      dataset_id: 'ds-2',
      value_column: 'reading',
      time_column: null,
      horizon: 24,
      frequency: 'D',
    })
  })

  it('连目标列都没有就返回 null，不吐半个请求体出去', () => {
    expect(payloadFromPreview({ id: 'ds-3' }, preview([['name', 'object', 5]]))).toBeNull()
  })
})

describe('latestSuccessfulTask + payloadFromTask', () => {
  const tasks = [
    { id: 'a', status: 'FAILED', dataset_id: 'd1', value_column: 'load', finished_at: '2026-09-11T09:00:00' },
    { id: 'b', status: 'SUCCESS', dataset_id: 'd2', value_column: 'load', time_column: 'hour_of_day', horizon: 24, frequency: 'high', finished_at: '2026-09-11T02:52:41' },
    { id: 'c', status: 'SUCCESS', dataset_id: 'd3', value_column: 'ac_power', time_column: 'days_since_start', horizon: 24, frequency: 'high', finished_at: '2026-09-11T03:28:57' },
  ]

  it('只挑成功的，并且按完成时间取最近一次', () => {
    expect(latestSuccessfulTask(tasks).id).toBe('c')
  })

  it('没有成功任务时返回 null', () => {
    expect(latestSuccessfulTask([tasks[0]])).toBeNull()
    expect(latestSuccessfulTask([])).toBeNull()
  })

  it('照抄成功任务的参数', () => {
    expect(payloadFromTask(tasks[1])).toEqual({
      dataset_id: 'd2',
      value_column: 'load',
      time_column: 'hour_of_day',
      horizon: 24,
      frequency: 'high',
    })
  })

  it('horizon 缺失或非法时回落到 24', () => {
    expect(payloadFromTask({ dataset_id: 'd', value_column: 'v', horizon: 0 }).horizon).toBe(24)
    expect(payloadFromTask({ dataset_id: 'd', value_column: 'v' }).horizon).toBe(24)
  })

  it('任务残缺就返回 null', () => {
    expect(payloadFromTask({ dataset_id: 'd' })).toBeNull()
    expect(payloadFromTask(null)).toBeNull()
  })
})

describe('missingTsFields', () => {
  it('只有 dataset_id 和 value_column 是必需的', () => {
    expect(missingTsFields({ dataset_id: 'd', value_column: 'load', time_column: null })).toEqual([])
  })

  it('列出缺的字段', () => {
    expect(missingTsFields({})).toEqual(['dataset_id', 'value_column'])
    expect(missingTsFields({ dataset_id: 'd' })).toEqual(['value_column'])
  })
})
